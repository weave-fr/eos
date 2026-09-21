"""Tests for the EOS dataset."""

import pytest
import torch

from eos.data import EOSTextDataset


def test_dataset_shapes():
    token_ids = list(range(100))

    dataset = EOSTextDataset(
        token_ids=token_ids,
        sequence_length=16,
    )

    sample = dataset[0]

    assert "input_ids" in sample
    assert "labels" in sample

    assert sample["input_ids"].shape == (16,)
    assert sample["labels"].shape == (16,)

    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long


def test_dataset_shifts_labels():
    token_ids = list(range(20))

    dataset = EOSTextDataset(
        token_ids=token_ids,
        sequence_length=8,
    )

    sample = dataset[0]

    assert sample["input_ids"].tolist() == list(range(8))
    assert sample["labels"].tolist() == list(range(1, 9))


def test_dataset_rejects_short_input():
    with pytest.raises(ValueError):
        EOSTextDataset(
            token_ids=[1, 2, 3],
            sequence_length=4,
        )
