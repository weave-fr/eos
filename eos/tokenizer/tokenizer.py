"""Tokenizer interface for EOS.

EOS 0.1 uses an existing tokenizer rather than creating
a proprietary tokenizer from scratch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


class EOSTokenizer:
    """Small wrapper around a Hugging Face-compatible tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name_or_path: str):
        """Load an existing tokenizer."""

        if not name_or_path:
            raise ValueError(
                "A tokenizer name or path must be provided."
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
        )

        return cls(tokenizer)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Convert text into token IDs."""

        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back to text."""

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""

        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID."""

        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID."""

        return self.tokenizer.eos_token_id

    def configure_model_vocab_size(
        self,
        model_config,
    ):
        """Set the model vocabulary size to match the tokenizer."""

        model_config.vocab_size = self.vocab_size

        return model_config

    def save_pretrained(
        self,
        path: str | Path,
    ) -> None:
        """Save the tokenizer."""

        self.tokenizer.save_pretrained(
            str(path),
        )
