"""Configuration objects for EOS models."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a decoder-only EOS Transformer."""

    vocab_size: int = 128_256
    context_length: int = 8_192

    hidden_size: int = 768
    num_layers: int = 24

    num_attention_heads: int = 12
    num_key_value_heads: int = 4

    intermediate_size: int = 2_048

    dropout: float = 0.0

    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True

    tie_word_embeddings: bool = True


@dataclass
class TinyModelConfig(ModelConfig):
    """Small configuration used to validate EOS before large training."""

    vocab_size: int = 32_000
    context_length: int = 1_024

    hidden_size: int = 256
    num_layers: int = 6

    num_attention_heads: int = 8
    num_key_value_heads: int = 2

    intermediate_size: int = 768
