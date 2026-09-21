"""Basic tests for the EOS model."""

import torch

from eos.model import EOSModel, TinyModelConfig


def test_eos_model_forward():
    """Verify that EOS can perform a forward pass."""

    config = TinyModelConfig()

    model = EOSModel(config)

    batch_size = 2
    sequence_length = 32

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
    )

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    assert "logits" in outputs
    assert "loss" in outputs

    assert outputs["logits"].shape == (
        batch_size,
        sequence_length,
        config.vocab_size,
    )

    assert torch.isfinite(outputs["loss"])
