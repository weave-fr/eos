"""Run a basic end-to-end test of EOS."""

from __future__ import annotations

import torch

from eos.model import EOSModel, TinyModelConfig


def main() -> None:
    print("=== EOS 0.1 — Model Test ===")

    config = TinyModelConfig()

    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Context length: {config.context_length}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers: {config.num_layers}")

    model = EOSModel(config)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")

    batch_size = 2
    sequence_length = 32

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
    )

    print("Running forward pass...")

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    logits = outputs["logits"]
    loss = outputs["loss"]

    if loss is None:
        raise RuntimeError("EOS did not return a loss.")

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.4f}")

    print("Running backward pass...")

    loss.backward()

    print("Backward pass: OK")
    print("EOS model test: SUCCESS")


if __name__ == "__main__":
    main()
