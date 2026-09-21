"""Training entry point for EOS."""

from __future__ import annotations

import torch

from eos.model import EOSModel, TinyModelConfig


def create_model() -> EOSModel:
    """Create the small EOS validation model."""

    config = TinyModelConfig()

    return EOSModel(config)


def training_step(
    model: EOSModel,
    input_ids: torch.Tensor,
) -> float:
    """Run one training step."""

    model.train()

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    loss = outputs["loss"]

    if loss is None:
        raise RuntimeError("Model did not return a loss.")

    loss.backward()

    return loss.item()
