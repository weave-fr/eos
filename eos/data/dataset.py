"""Dataset utilities for EOS training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EOSTextDataset(Dataset):
    """Simple tokenized text dataset for causal language modeling."""

    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2"
            )

        if len(token_ids) < sequence_length + 1:
            raise ValueError(
                "Not enough tokens for one training example"
            )

        self.token_ids = token_ids
        self.sequence_length = sequence_length

        self.num_examples = (
            len(token_ids) - 1
        ) // sequence_length

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        start = index * self.sequence_length
        end = start + self.sequence_length + 1

        tokens = self.token_ids[start:end]

        input_ids = torch.tensor(
            tokens[:-1],
            dtype=torch.long,
        )

        labels = torch.tensor(
            tokens[1:],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def load_token_ids(path: str | Path) -> list[int]:
    """Load a simple token ID file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    return [
        int(token)
        for token in text.split()
    ]
