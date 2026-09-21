"""Core Transformer implementation for EOS."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, inv_freq)

        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        cos = self.cos[:seq_len].to(dtype=x.dtype)
        sin = self.sin[:seq_len].to(dtype=x.dtype)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings."""

    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        return self.down_proj(gate * value)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.head_dim = (
            config.hidden_size // config.num_attention_heads
        )

        self.num_kv_groups = (
            self.num_heads // self.num_kv_heads
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.context_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = self.rope(q, seq_len)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(
                self.num_kv_groups,
                dim=1,
            )

            v = v.repeat_interleave(
                self.num_kv_groups,
                dim=1,
            )

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        attention = attention.transpose(1, 2).contiguous()

        attention = attention.view(
            batch_size,
            seq_len,
            -1,
        )

        return self.o_proj(attention)


class TransformerBlock(nn.Module):
    """Single EOS Transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.input_norm = RMSNorm(config.hidden_size)

        self.attention = GroupedQueryAttention(config)

        self.post_attention_norm = RMSNorm(
            config.hidden_size
        )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(
            self.input_norm(x)
        )

        x = x + self.mlp(
            self.post_attention_norm(x)
        )

        return x


class EOSModel(nn.Module):
    """EOS decoder-only Transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.hidden_size
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        output = {
            "logits": logits,
        }

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            output["loss"] = loss

        return output
