import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def conv1d(x, conv):
    # print(conv.kernel_size, conv.stride, conv.padding)
    conv2d = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        (1, conv.kernel_size[0]),
        (1, conv.stride[0]),
        padding=(0, conv.padding[0]),
        bias=conv.bias is not None,
    )
    conv2d.weight = nn.Parameter(conv.weight[:, :, np.newaxis])
    if conv.bias is not None:
        conv2d.bias = nn.Parameter(conv.bias)

    return conv2d(x[:, :, np.newaxis])[:, :, 0]


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, decoder: bool = False):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        # n_batch, n_ctx, n_state = q.shape
        n_batch, n_ctx, n_state = 1, 1500, 512
        # shape = [1, 1500, 512]
        # print(n_batch, n_ctx, n_state)
        scale = (n_state // self.n_head) ** -0.25
        # q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        # k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        # v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        q = q.view(1, 1500, self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(1, 1500, self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(1, 1500, self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class DecoderMultiHeadCrossAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)

        k = key_cache
        v = value_cache
        # print("cache", k.shape, v.shape)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        # print(q.shape)
        # n_batch, n_ctx, n_state = q.shape
        n_batch, n_ctx, n_state = 1, 1, 512
        scale = (n_state // self.n_head) ** -0.25
        # print("cross", q.shape, k.shape, v.shape)
        # q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        # k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        # v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        q = q.view(1, 1, self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(1, 1500, self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(1, 1500, self.n_head, -1).permute(0, 2, 1, 3)
        # print("dm shape", q.shape, k.shape, v.shape)
        # q = q.view(1, 1, self.n_head, -1).permute(0, 2, 1, 3) * scale
        # k = k.view(1, 1500, self.n_head, -1).permute(0, 2, 3, 1) * scale
        # v = v.view(1, 1500, self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class DecoderMultiHeadSelfAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        k = torch.cat([key_cache, k], dim=1).detach()
        v = torch.cat([value_cache, v], dim=1).detach()
        # print("dmsa", q.shape, k.shape, v.shape)
        # wv, qk = self.qkv_attention(q, k[:, 1 : k.shape[1]], v[:, 1 : v.shape[1]], mask)
        # wv, qk = self.qkv_attention(q, k[:, 1 : k.shape[1]], v[:, 1 : v.shape[1]], mask)
        wv, qk = self.qkv_attention(q, k, v, mask)

        return self.out(wv), qk, k, v

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        # n_batch, n_ctx, n_state = q.shape
        n_batch, n_ctx, n_state = 1, 1, 512
        # print(n_batch, n_ctx, n_state)
        scale = (n_state // self.n_head) ** -0.25
        # print("qkv", q.shape, k.shape, v.shape)
        # q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        # k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        # v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        q = q.view(1, 1, self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class GELU(nn.Module):
    def forward(self, x: Tensor):
        return (
            0.5
            * x
            * (1 + torch.tanh(np.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
        )


def layer_norm(x, ln):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True, unbiased=False)
    norm = (x - mean) / (std + ln.eps)
    out = ln.weight * norm + ln.bias
    return out


class EncoderResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
    ):
        super().__init__()
        self.attn = EncoderMultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        # print("ln", self.attn_ln.weight, self.attn_ln.bias, self.attn_ln.eps)
        # x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.attn(layer_norm(x, self.attn_ln), mask=mask)[0]
        # x = x + self.mlp(self.mlp_ln(x))
        x = x + self.mlp(layer_norm(x, self.mlp_ln))
        return x


class DecoderResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
    ):
        super().__init__()
        self.attn = DecoderMultiHeadSelfAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = DecoderMultiHeadCrossAttention(n_state, n_head)
        self.cross_attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        self_attn_key_cache: Tensor,
        self_attn_value_cache: Tensor,
        cross_attn_key_cache: Tensor,
        cross_attn_value_cache: Tensor,
        mask: Optional[Tensor] = None,
    ):
        # print("pre self attn", x.shape)
        # print("cross key", cross_attn_key_cache.shape)
        # print("cross value", cross_attn_value_cache.shape)
        wv, _, self_k, self_v = self.attn(
            # self.attn_ln(x),
            layer_norm(x, self.attn_ln),
            self_attn_key_cache,
            self_attn_value_cache,
            mask=mask,
        )
        # print("post self attn", wv.shape, self_k.shape, self_v.shape)
        x = x + wv
        x = (
            x
            + self.cross_attn(
                # self.cross_attn_ln(x),
                layer_norm(x, self.cross_attn_ln),
                cross_attn_key_cache,
                cross_attn_value_cache,
            )[0]
        )
        # x = x + self.mlp(self.mlp_ln(x))
        x = x + self.mlp(layer_norm(x, self.mlp_ln))
        return x, self_k, self_v


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[EncoderResidualAttentionBlock] = nn.ModuleList(
            [EncoderResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # x = GELU()(self.conv1(x))
        # x = GELU()(self.conv2(x))
        x = GELU()(conv1d(x, self.conv1))
        x = GELU()(conv1d(x, self.conv2))
        x = x.permute(0, 2, 1)

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        # x = self.ln_post(x)
        x = layer_norm(x, self.ln_post)
        return x


class AudioEncoderWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.cross_attns = []
        for block in decoder.blocks:
            self.cross_attns.append(block.cross_attn)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        cross_attn_keys = []
        cross_attn_values = []
        for cross_attn in self.cross_attns:
            cross_attn_keys.append(cross_attn.key(x))
            cross_attn_values.append(cross_attn.value(x))
        cross_attn_keys = torch.cat(cross_attn_keys, dim=0)
        cross_attn_values = torch.cat(cross_attn_values, dim=0)
        return x, cross_attn_keys, cross_attn_values


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[DecoderResidualAttentionBlock] = nn.ModuleList(
            [DecoderResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        self_attn_key_cache,
        self_attn_value_cache,
        cross_attn_key_cache,
        cross_attn_value_cache,
        positional_embedding,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        print("dec")
        print("token", x)
        print("self key", self_attn_key_cache)
        print("self val", self_attn_value_cache)
        print("cross key", cross_attn_key_cache)
        print("cross val", cross_attn_value_cache)
        print("pos emb", positional_embedding)
        # offset = self_attn_value_cache.shape[1] - 1
        # print("offset", offset)
        # print("pre embed", x.shape)
        x = (
            self.token_embedding(x)
            + positional_embedding
            # + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        # print("post embed", x.shape)
        # x = x.to(xa.dtype)

        self_attn_keys = []
        self_attn_values = []
        # print("pre dec", self_attn_value_cache[:1].shape)
        # print("self cache", self_attn_key_cache.shape, self_attn_value_cache.shape)
        # print("cross cache", cross_attn_key_cache.shape, cross_attn_value_cache.shape)
        for n, block in enumerate(self.blocks):
            x, k, v = block(
                x,
                self_attn_key_cache[n : n + 1],
                self_attn_value_cache[n : n + 1],
                cross_attn_key_cache[n : n + 1],
                cross_attn_value_cache[n : n + 1],
                mask=self.mask,
            )
            self_attn_keys.append(k)
            self_attn_values.append(v)
        # print("post dec", self_attn_values[0].shape)
        self_attn_keys = torch.cat(self_attn_keys, dim=0)
        self_attn_values = torch.cat(self_attn_values, dim=0)

        # x = self.ln(x)
        x = layer_norm(x, self.ln)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, self_attn_keys, self_attn_values


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # self.encoder = AudioEncoderWrapper(self.encoder, self.decoder)
        self.self_attn_key_cache = torch.zeros(len(self.decoder.blocks), 1, 512).to(
            "cuda"
        )
        self.self_attn_value_cache = torch.zeros(len(self.decoder.blocks), 1, 512).to(
            "cuda"
        )
        self.cross_attn_key_cache = torch.zeros(len(self.decoder.blocks), 1, 512).to(
            "cuda"
        )
        self.cross_attn_value_cache = torch.zeros(len(self.decoder.blocks), 1, 512).to(
            "cuda"
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(
        self,
        tokens: torch.Tensor,
    ):
        offset = self.self_attn_value_cache.shape[1] - 1
        positional_embedding = self.decoder.positional_embedding[
            offset : offset + tokens.shape[-1]
        ]
        l, new_self_attn_key_cache, new_self_attn_value_cache = self.decoder(
            tokens,
            self.self_attn_key_cache,
            self.self_attn_value_cache,
            self.cross_attn_key_cache,
            self.cross_attn_value_cache,
            positional_embedding,
        )
        # print("logits, token", tokens.shape)
        # print("pre cache", self.model.self_attn_value_cache.shape)
        self.self_attn_key_cache = new_self_attn_key_cache
        self.self_attn_value_cache = new_self_attn_value_cache
        # print("post cache", self.model.self_attn_value_cache.shape)
        return l

    def detection_logits(
        self,
        tokens: torch.Tensor,
    ):
        offset = self.self_attn_value_cache.shape[1] - 1
        positional_embedding = self.decoder.positional_embedding[
            offset : offset + tokens.shape[-1]
        ]
        l, new_self_attn_key_cache, new_self_attn_value_cache = self.decoder(
            tokens,
            self.self_attn_key_cache,
            self.self_attn_value_cache,
            self.cross_attn_key_cache,
            self.cross_attn_value_cache,
            positional_embedding,
        )
        print("detection_logits", tokens, l)
        # self.self_attn_key_cache = new_self_attn_key_cache
        # self.self_attn_value_cache = new_self_attn_value_cache
        return l

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        detect_language: torch.Tensor = torch.zeros(1, dtype=torch.bool),
    ) -> Dict[str, torch.Tensor]:
        audio_features, cross_attn_keys, cross_attn_values = self.encoder(mel)
        self.cross_attn_key_cache = cross_attn_keys
        self.cross_attn_value_cache = cross_attn_values
        if detect_language.all():
            (
                l,
                new_self_attn_key_cache,
                new_self_attn_value_cache,
            ) = self.decoder.detection(
                tokens,
                audio_features,
                self.self_attn_key_cache,
                self.self_attn_value_cache,
                self.cross_attn_key_cache,
                self.cross_attn_value_cache,
            )
        else:
            l, new_self_attn_key_cache, new_self_attn_value_cache = self.decoder(
                tokens,
                audio_features,
                self.self_attn_key_cache,
                self.self_attn_value_cache,
                self.cross_attn_key_cache,
                self.cross_attn_value_cache,
            )
        # print("detection_logits")
        # print("pre cache", self.model.self_attn_value_cache.shape)
        self.self_attn_key_cache = new_self_attn_key_cache
        self.self_attn_value_cache = new_self_attn_value_cache
        # print("post cache", self.model.self_attn_value_cache.shape)
        # self.cross_attn_key_cache
        return l

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    # def install_kv_cache_hooks(self, cache: Optional[dict] = None):
    # """
    # The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
    # tensors calculated for the previous positions. This method returns a dictionary that stores
    # all caches, and the necessary hooks for the key and value projection modules that save the
    # intermediate tensors to be reused during later calculations.

    # Returns
    # -------
    # cache : Dict[nn.Module, torch.Tensor]
    # A dictionary object mapping the key/value projection modules to its cache
    # hooks : List[RemovableHandle]
    # List of PyTorch RemovableHandle objects to stop the hooks to be called
    # """
    # cache = {**cache} if cache is not None else {}
    # hooks = []

    # def save_to_cache(module, _, output):
    # idx = (
    # self.cross_attn_to_idx[module]
    # if module.cross
    # else self.self_attn_to_idx[module]
    # )
    # if module not in cache or output.shape[1] > self.dims.n_text_ctx:
    # # save as-is, for the first token or cross attention
    # cache[module] = output
    # else:
    # cache[module] = torch.cat([cache[module], output], dim=1).detach()
    # return cache[module]

    # def install_hooks(layer: nn.Module):
    # if isinstance(layer, MultiHeadAttention):
    # hooks.append(layer.key.register_forward_hook(save_to_cache))
    # hooks.append(layer.value.register_forward_hook(save_to_cache))

    # self.decoder.apply(install_hooks)
    # return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
