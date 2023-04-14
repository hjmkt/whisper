import onnx
import torch
import numpy as np
import whisper
import argparse
import onnxruntime as ort
from whisper.tokenizer import get_tokenizer

SAMPLE_RATE = 16000


class WhisperWrapper:
    def __init__(self, dims, n_blocks, positional_embedding, chunk_length):
        sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = (
        # ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # )
        self.ort_session_encoder = ort.InferenceSession(
            "models/encoder_float32.onnx",
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.ort_session_decoder = ort.InferenceSession(
            "models/decoder_float32.onnx",
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.ort_session_preprocessor = ort.InferenceSession(
            "models/preprocessor_float32.onnx",
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.dims = dims
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.self_attn_key_cache = torch.zeros(n_blocks, 1, 512).to(device)
        self.self_attn_value_cache = torch.zeros(n_blocks, 1, 512).to(device)
        self.cross_attn_key_cache = torch.zeros(n_blocks, chunk_length * 50, 512).to(
            device
        )
        self.cross_attn_value_cache = torch.zeros(n_blocks, chunk_length * 50, 512).to(
            device
        )
        self.positional_embedding = positional_embedding

    def encoder(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        output = self.ort_session_encoder.run(None, {"input": x.numpy()})
        self.cross_attn_key_cache = torch.from_numpy(output[1])
        self.cross_attn_value_cache = torch.from_numpy(output[2])
        return torch.from_numpy(output[0])

    def decoder(
        self,
        tokens,
        self_attn_key_cache,
        self_attn_value_cache,
        cross_attn_key_cache,
        cross_value_cache,
        positional_embedding,
    ):
        output = self.ort_session_decoder.run(
            None,
            {
                "input_token": tokens.numpy().astype(np.int32),
                "self_key_cache": self.self_attn_key_cache.cpu().numpy(),
                "self_value_cache": self.self_attn_value_cache.cpu().numpy(),
                "cross_key_cache": self.cross_attn_key_cache.numpy(),
                "cross_value_cache": self.cross_attn_value_cache.numpy(),
                "positional_embedding": positional_embedding.cpu().numpy(),
            },
        )
        l, k, v = output
        return torch.from_numpy(l), torch.from_numpy(k), torch.from_numpy(v)

    def preprocessor(self, x):
        output = self.ort_session_preprocessor.run(None, {"input": x})
        return torch.from_numpy(output[0])

    def logits(self, tokens):
        offset = self.self_attn_value_cache.shape[1] - 1
        positional_embedding = self.positional_embedding[
            offset : offset + tokens.shape[-1]
        ]
        l, k, v = self.decoder(
            tokens,
            self.self_attn_key_cache,
            self.self_attn_value_cache,
            self.cross_attn_key_cache,
            self.cross_attn_value_cache,
            positional_embedding,
        )
        self.self_attn_key_cache = k
        self.self_attn_value_cache = v
        return l

    def detection_logits(self, tokens):
        offset = self.self_attn_value_cache.shape[1] - 1
        positional_embedding = self.positional_embedding[
            offset : offset + tokens.shape[-1]
        ]
        l, k, v = self.decoder(
            tokens,
            self.self_attn_key_cache,
            self.self_attn_value_cache,
            self.cross_attn_key_cache,
            self.cross_attn_value_cache,
            positional_embedding,
        )
        return l

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def detect_language(self, mel, tokenizer):
        if tokenizer is None:
            tokenizer = get_tokenizer(self.is_multilingual)
        if (
            tokenizer.language is None
            or tokenizer.language_token not in tokenizer.sot_sequence
        ):
            raise ValueError(
                "This model doesn't have language tokens so it can't perform lang id"
            )

        single = mel.ndim == 2
        if single:
            mel = mel.unsqueeze(0)

        # skip encoder forward pass if already-encoded audio features were given
        if mel.shape[-2:] != (self.dims.n_audio_ctx, self.dims.n_audio_state):
            mel = self.encoder(mel)

        # forward pass using a single token, startoftranscript
        n_audio = mel.shape[0]
        x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
        logits = self.detection_logits(x)[:, 0]

        # collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(
                    tokenizer.all_language_tokens, tokenizer.all_language_codes
                )
            }
            for i in range(n_audio)
        ]

        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        return language_tokens, language_probs


def run(chunk_length):
    torch_model = whisper.load_model("base", chunk_length=chunk_length)
    model_dims = torch_model.dims

    # print(onnx.helper.printable_graph(model.graph))
    positional_embedding = torch_model.decoder.positional_embedding
    wrapper = WhisperWrapper(model_dims, 6, positional_embedding, chunk_length)

    audio = whisper.load_audio("/home/hjmkt/Documents/ja.mp3")
    audio = whisper.pad_or_trim(audio, chunk_length * SAMPLE_RATE)

    mel = wrapper.preprocessor(audio)

    options = whisper.DecodingOptions()
    result = whisper.decode(wrapper, mel, options)

    print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ONNX Whisper Test", description="Test ONNX Whisper model"
    )
    parser.add_argument("--chunk_length", type=int, default=30)
    args = parser.parse_args()
    run(**vars(args))
