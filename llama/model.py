# Copyright (c) Advanced AI Labs. All rights reserved.
# This software is licensed under the Advanced Transformer Research License.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from einops import rearrange, repeat

# Optional: For advanced attention efficiency (assuming torch >= 2.0)
# No external dependencies like Fairscale; using native PyTorch distributed if needed


@dataclass
class AdvancedModelConfig:
    vocab_size: int = 50257
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: Optional[int] = None  # For GQA/MQA
    head_dim: Optional[int] = None
    intermediate_dim: int = 8192
    activation: str = "gelu"  # Options: gelu, swiglu, relu
    norm_type: str = "layernorm"  # Options: layernorm, rmsnorm
    dropout: float = 0.1
    max_position_embeddings: int = 4096
    rope_scaling: Optional[dict] = None  # For dynamic NTK/Yarn scaling
    use_flash_attention: bool = True  # Use scaled_dot_product_attention if available
    use_moe: bool = True  # Mixture of Experts in FFN
    num_experts: int = 8
    moe_top_k: int = 2
    use_bias: bool = False
    tie_word_embeddings: bool = False
    max_batch_size: int = 64
    max_seq_len: int = 8192
    dtype: torch.dtype = torch.bfloat16


class LayerNorm(nn.Module):
    """Advanced LayerNorm with optional bias and learnable scale."""
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias = Parameter(torch.zeros(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor):
        output = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
        if self.weight is not None:
            output = output * self.weight + self.bias
        return output


class RMSNorm(nn.Module):
    """RMSNorm variant with optional affine transform."""
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        x_normed = x / (norm + self.eps)
        if self.scale is not None:
            x_normed = x_normed * self.scale
        return x_normed


class RotaryEmbedding(nn.Module):
    """Advanced RoPE with optional scaling (NTK or Yarn)."""
    def __init__(self, dim: int, max_position: int = 8192, base: float = 10000.0, scaling_type: str = "linear"):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scaling_type = scaling_type
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self):
        t = torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.inv_freq.device)
        return 1.0 / (self.base ** (t / self.dim))

    def forward(self, positions: torch.Tensor, seq_len: int):
        freqs = torch.einsum("i, j -> i j", self.inv_freq, positions)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.scaling_type == "yarn":
            scale = self._compute_yarn_scale(seq_len)
            emb = emb * scale.unsqueeze(0)
        return torch.view_as_complex(torch.exp(1j * emb))

    def _compute_yarn_scale(self, seq_len: int):
        # Simple Yarn-inspired scaling
        return torch.tensor(1.0 / math.log(seq_len / self.base + 1), device=self.inv_freq.device)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to queries and keys."""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rot(tensors, cos, sin):
        return [ro * c + im * s for ro, im, c, s in zip(rearrange(tensors, "b ... (d r) -> ... b d r", r=2), cos, sin)]

    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)
    q_embed = apply_rot(q, cos, sin)
    k_embed = apply_rot(k, cos, sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Advanced MHA with GQA/MQA support, FlashAttention, and KV caching."""
    def __init__(self, config: AdvancedModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.head_dim or (config.hidden_dim // config.num_heads)
        self.query = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.output = nn.Linear(self.num_heads * self.head_dim, config.hidden_dim, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

        # KV Cache
        self.register_buffer("k_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        bsz, tgt_len, _ = x.shape

        # Compute Q, K, V
        query_states = self.query(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.key(x).view(bsz, tgt_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.value(x).view(bsz, tgt_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(torch.arange(start_pos, start_pos + tgt_len, device=x.device), tgt_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV Cache
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        else:
            key_states = key_states
            value_states = value_states

        # Repeat KV for GQA/MQA
        if self.num_kv_heads != self.num_heads:
            key_states = repeat(key_states, "b hkv d t -> b (hkv hk) d t", hk=(self.num_heads // self.num_kv_heads))
            value_states = repeat(value_states, "b hkv d t -> b (hkv hk) d t", hk=(self.num_heads // self.num_kv_heads))

        # Attention with Flash if enabled
        if self.config.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights += attention_mask
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
        attn_output = self.output(attn_output)

        return attn_output, (key_states, value_states)


class MoEFFN(nn.Module):
    """Mixture of Experts FeedForward with top-k routing."""
    def __init__(self, config: AdvancedModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.intermediate_dim, bias=config.use_bias),
                nn.GELU() if config.activation == "gelu" else nn.SiLU() if "swi" in config.activation else nn.ReLU(),
                nn.Linear(self.intermediate_dim, self.hidden_dim, bias=config.use_bias),
            ) for _ in range(self.num_experts)
        ])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_dim)
        gate_logits = self.gate(x_flat)  # [bsz*seq_len, num_experts]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(-1)
            for j, expert in enumerate(self.experts):
                mask = (expert_idx == j).unsqueeze(-1).float()
                expert_out = expert(x_flat)
                output += mask * expert_out * weight
        output = self.dropout(output)
        return output.view(bsz, seq_len, self.hidden_dim)


class FFN(nn.Module):
    """Standard FFN fallback if MoE disabled."""
    def __init__(self, config: AdvancedModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=config.use_bias)
        self.act = nn.GELU() if config.activation == "gelu" else nn.SiLU() if "swi" in config.activation else nn.ReLU()
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class AdvancedTransformerBlock(nn.Module):
    """Advanced Transformer Block with pre-norm, GQA, and MoE."""
    def __init__(self, config: AdvancedModelConfig, layer_idx: int):
        super().__init__()
        norm_class = LayerNorm if config.norm_type == "layernorm" else RMSNorm
        self.norm1 = norm_class(config.hidden_dim, elementwise_affine=True)
        self.attn = MultiHeadAttention(config)
        self.norm2 = norm_class(config.hidden_dim, elementwise_affine=True)
        self.ffn = MoEFFN(config) if config.use_moe else FFN(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        layer_past: Optional[List[Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        residual = x
        x = self.norm1(x)
        attn_output, new_past = self.attn(x, attention_mask, start_pos, layer_past[0] if layer_past else None)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x, [new_past]


class AdvancedTransformer(nn.Module):
    """Advanced Decoder-Only Transformer with RoPE, GQA, MoE, and FlashAttention."""
    def __init__(self, config: AdvancedModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(config, i) for i in range(config.num_layers)
        ])
        self.norm = LayerNorm(config.hidden_dim, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.rope = RotaryEmbedding(config.head_dim or (config.hidden_dim // config.num_heads), config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)

        # Compile for speed (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode="reduce-overhead")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        # Positional embeddings via RoPE (applied in attention)

        # Attention mask
        if attention_mask is None:
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=self.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attention_mask = causal_mask if seq_len > 1 else None

        # Extend mask for past KV
        if past_key_values is not None and seq_len > 1:
            past_len = past_key_values[0][0].shape[2]
            mask_shape = (seq_len, past_len + seq_len)
            mask = torch.zeros(mask_shape, device=device, dtype=self.dtype)
            mask[:, -seq_len:] = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=self.dtype)
            mask = torch.triu(mask, diagonal=-past_len)
            attention_mask = mask

        # Forward layers
        presents = () if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = (past_key_values[i][0], past_key_values[i][1]) if past_key_values is not None else None
            x, new_past = layer(x, attention_mask, 0, layer_past)
            if use_cache:
                presents = presents + (new_past,)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, presents
