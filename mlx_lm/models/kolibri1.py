# Copyright © 2026 Apple Inc.

# Reference: https://github.com/Aleph-Alpha/aleph-alpha-inference (kolibri1.py)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    layer_types: List[str] = field(default_factory=list)
    vocab_size: int = 128000
    hidden_size: int = 2560
    num_hidden_layers: int = 50
    num_attention_heads: int = 48
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    num_experts: int = 384
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = False
    # Includes the query token (512 preceding + 1).
    sliding_window: int = 513


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = (
            initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,
                args.rope_scaling,
                args.max_position_embeddings,
            )
            if use_sliding
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Kolibri1MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        if isinstance(self.gate, nn.QuantizedLinear):
            logits = self.gate(x).astype(mx.float32)
        else:
            logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T

        # The bias changes the selection only, not the weights.
        k = self.num_experts_per_tok
        selection = logits + self.expert_bias
        inds = mx.stop_gradient(
            mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.sigmoid(mx.take_along_axis(logits, inds, axis=-1))
        if self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

        y = self.experts(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Attention(args, use_sliding)
        self.mlp = Kolibri1MoE(args)

        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attn_norm(r)
        r = self.mlp(self.post_attention_layernorm(h))
        return h + self.post_ffn_norm(r)


class Kolibri1Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, use_sliding=t == "sliding_attention")
            for t in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = args.layer_types.index("full_attention")
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            bs = 128
            m, n = weight.shape
            pad_bottom = bs * scale_inv.shape[0] - m
            pad_side = bs * scale_inv.shape[1] - n
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(mx.bfloat16)

        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # The official release is block-FP8 (128x128, fp32 inverse scales).
        for sk in [k for k in weights if k.endswith(".weight_scale_inv")]:
            wk = sk[: -len("_scale_inv")]
            weights[wk] = dequant(weights[wk], weights.pop(sk))

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            bias_key = f"{prefix}.moe.router.expert_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.expert_bias"] = weights.pop(bias_key)
            for n in ["gate_proj", "up_proj", "down_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.experts.{n}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        # The release keeps the router in bf16.
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
