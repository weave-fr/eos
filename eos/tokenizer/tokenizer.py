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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path)

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
        """Convert token IDs back into text."""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return len(self.tokenizer)

    def save_pretrained(self, path: str | Path) -> None:
        """Save the tokenizer."""
        self.tokenizer.save_pretrained(str(path))
