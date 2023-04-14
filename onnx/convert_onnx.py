import torch
import whisper
import librosa
import gzip
import os
import sys
import argparse
import numpy as np
from whisper.model import AudioEncoderWrapper
from torch import Tensor, nn
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160


class STFT(torch.nn.Module):
    """
    Simulate torch.stft by using a convolution layer
    as a workaround for ONNX's limited layer support
    """

    def __init__(self, n_fft, hop_length, window_periodic=True):
        super().__init__()
        self.win_length = n_fft
        self.hop_length = hop_length
        self.nfft = n_fft
        self.freq_cutoff = self.nfft // 2 + 1
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length, periodic=window_periodic).float(),
        )
        fourier_basis = torch.fft.fft(torch.eye(self.nfft))
        fourier_basis = torch.view_as_real(fourier_basis)
        forward_basis = (
            fourier_basis[: self.freq_cutoff]
            .permute(2, 0, 1)
            .reshape(-1, 1, fourier_basis.shape[1])
        )
        forward_basis = forward_basis * torch.as_tensor(
            librosa.util.pad_center(self.window, size=self.nfft),
            dtype=forward_basis.dtype,
        )

        forward_basis = forward_basis.reshape(
            forward_basis.shape[0],
            forward_basis.shape[1],
            1,
            forward_basis.shape[2],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simulate Conv1d by using Conv2d as a workaround for ONNX's limited layer support
        self.stft = (
            torch.nn.Conv2d(
                forward_basis.shape[1],
                forward_basis.shape[0],
                forward_basis.shape[2:4],
                bias=False,
                stride=(1, self.hop_length),
            )
            .to(device)
            .requires_grad_(False)
        )
        self.stft.weight.copy_(forward_basis)

    def forward(self, signal):
        pad = self.freq_cutoff - 1
        signal = torch.unsqueeze(signal, 0).unsqueeze(1)

        # padded_signal = torch.nn.functional.pad(signal, (pad, pad), mode="reflect").squeeze(1)
        # Simulate the above padding by concatenating zero tensors
        # as a workaround for ONNX's limited layer support
        # The simulated padding mode is "constant" instead of "reflect",
        # which doesn't much affect the result
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pad_zeros = torch.zeros(signal.shape[0], signal.shape[1], pad).to(device)
        padded_signal = torch.cat((pad_zeros, signal, pad_zeros), dim=2)

        stft_signal = self.stft(padded_signal.unsqueeze(dim=1))
        real = stft_signal[:, : self.freq_cutoff, 0, :]
        imag = stft_signal[:, self.freq_cutoff : self.freq_cutoff * 2, 0, :]

        return real, imag


def mel_filters(mel_filters_path, device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def relu_max(x, max):
    """
    Simulate Max by using ReLU as a workaround for ONNX's limited layer support
    """
    return nn.ReLU()(x - max) + max


class LogMelSpectrogram(nn.Module):
    def __init__(self, mel_filters_path):
        super().__init__()
        self.stft = STFT(N_FFT, HOP_LENGTH)
        self.mel_filters_path = mel_filters_path

    def forward(self, x: Tensor) -> Tensor:
        audio = x
        n_mels = N_MELS
        padding = 0
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        if padding > 0:
            audio = F.pad(audio, (0, padding))

        stft_real, stft_imag = self.stft(audio)
        magnitudes = (stft_real[..., :-1] ** 2 + stft_imag[..., :-1] ** 2)[0]

        filters = mel_filters(self.mel_filters_path, audio.device, n_mels)
        mel_spec = filters @ magnitudes

        log_spec = relu_max(mel_spec, 1e-10).log10()
        log_spec = relu_max(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def convert(
    output_dir="models",
    mel_filters_path="mel_filters.npz",
    model_type="base",
    opset_version=17,
    chunk_length=30,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = whisper.load_model(model_type, chunk_length=chunk_length)
    model.encoder = AudioEncoderWrapper(model.encoder, model.decoder)
    preprocessor = LogMelSpectrogram(mel_filters_path)

    positional_embedding = model.decoder.positional_embedding.cpu().detach().numpy()
    print("pe shape", positional_embedding.shape, positional_embedding.dtype)
    with gzip.open(f"{output_dir}/positional_embedding.bin.gz", "wb") as f:
        f.write(positional_embedding.tobytes())
        # print(
        # "export let positional_embedding =",
        # list([list(p) for p in positional_embedding]),
        # ";",
        # file=f,
        # )

    def convert_long_to_int(model):
        for name, param in model.named_parameters():
            if param.dtype == torch.int64:
                param.data = param.data.to(torch.int32)
        for name, buffer in model.named_buffers():
            if buffer.dtype == torch.int64:
                buffer.data = buffer.data.to(torch.int32)

        for child in model.children():
            convert_long_to_int(child)

    convert_long_to_int(model.encoder)
    convert_long_to_int(model.decoder)
    convert_long_to_int(preprocessor)

    model.encoder.parameters()
    model.decoder.parameters()
    preprocessor.parameters()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in preprocessor.parameters():
        p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_input_encoder = torch.randn(
        1,
        80,
        chunk_length * 100,
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    dummy_input_decoder_token = (
        (torch.randn(1, 1, requires_grad=False) * 10).int().to(device)
    )
    dummy_input_decoder_self_key_cache = torch.randn(
        len(model.decoder.blocks), 1, 512, dtype=torch.float32, requires_grad=False
    ).to(device)
    dummy_input_decoder_self_value_cache = torch.randn(
        len(model.decoder.blocks), 1, 512, dtype=torch.float32, requires_grad=False
    ).to(device)
    dummy_input_decoder_cross_key_cache = torch.randn(
        len(model.decoder.blocks),
        chunk_length * 50,
        512,
        dtype=torch.float32,
        requires_grad=False,
    ).to(device)
    dummy_input_decoder_cross_value_cache = torch.randn(
        len(model.decoder.blocks),
        chunk_length * 50,
        512,
        dtype=torch.float32,
        requires_grad=False,
    ).to(device)
    dummy_input_decoder_positional_embedding = torch.randn(
        1, 512, dtype=torch.float32, requires_grad=False
    ).to(device)
    dummy_input_preprocessor = torch.randn(
        SAMPLE_RATE * chunk_length,
        dtype=torch.float32,
        requires_grad=False,
    ).to(device)

    torch.onnx.export(
        preprocessor,
        dummy_input_preprocessor,
        f"{output_dir}/preprocessor_{chunk_length}s_float32.onnx",
        # verbose=True,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=opset_version,
    )
    torch.onnx.export(
        model.encoder,
        dummy_input_encoder,
        f"{output_dir}/encoder_{chunk_length}s_float32.onnx",
        input_names=["input"],
        output_names=["output", "key_cache", "value_cache"],
        opset_version=opset_version,
    )
    torch.onnx.export(
        model.decoder,
        args=(
            dummy_input_decoder_token,
            dummy_input_decoder_self_key_cache,
            dummy_input_decoder_self_value_cache,
            dummy_input_decoder_cross_key_cache,
            dummy_input_decoder_cross_value_cache,
            dummy_input_decoder_positional_embedding,
        ),
        f=f"{output_dir}/decoder_{chunk_length}s_float32.onnx",
        input_names=[
            "input_token",
            "self_key_cache",
            "self_value_cache",
            "cross_key_cache",
            "cross_value_cache",
            "positional_embedding",
        ],
        output_names=["output", "key_cache", "value_cache"],
        dynamic_axes={
            "input_token": [1],
            "self_key_cache": [1],
            "self_value_cache": [1],
            "cross_key_cache": [1],
            "cross_value_cache": [1],
            "positional_embedding": [0],
        },
        opset_version=16,
    )
    print("Quantizing preprocessor...", file=sys.stderr)
    quantize_dynamic(
        f"{output_dir}/preprocessor_{chunk_length}s_float32.onnx",
        f"{output_dir}/preprocessor_{chunk_length}s_int8.onnx",
        weight_type=QuantType.QUInt8,
    )
    print("Quantizing encoder...", file=sys.stderr)
    quantize_dynamic(
        f"{output_dir}/encoder_{chunk_length}s_float32.onnx",
        f"{output_dir}/encoder_{chunk_length}s_int8.onnx",
        weight_type=QuantType.QUInt8,
    )
    print("Quantizing decoder...", file=sys.stderr)
    quantize_dynamic(
        f"{output_dir}/decoder_{chunk_length}s_float32.onnx",
        f"{output_dir}/decoder_{chunk_length}s_int8.onnx",
        weight_type=QuantType.QUInt8,
    )

    def gzip_onnx(onnx_file):
        print(f"GZIP-compressing {onnx_file}...", file=sys.stderr)
        with open(onnx_file, "rb") as f:
            data = f.read()
            with gzip.open(f"{onnx_file}.gz", "wb") as f:
                f.write(data)

    gzip_onnx(f"{output_dir}/preprocessor_{chunk_length}s_float32.onnx")
    gzip_onnx(f"{output_dir}/encoder_{chunk_length}s_float32.onnx")
    gzip_onnx(f"{output_dir}/decoder_{chunk_length}s_float32.onnx")
    gzip_onnx(f"{output_dir}/preprocessor_{chunk_length}s_int8.onnx")
    gzip_onnx(f"{output_dir}/encoder_{chunk_length}s_int8.onnx")
    gzip_onnx(f"{output_dir}/decoder_{chunk_length}s_int8.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ONNX Converter", description="Convert PyTorch Whisper model to ONNX"
    )
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--mel_filters_path", type=str)
    parser.add_argument("--model_type", type=str, default="base")
    parser.add_argument("--opset_version", type=int, default=17)
    parser.add_argument("--chunk_length", type=int, default=30)
    args = parser.parse_args()
    convert(**vars(args))
