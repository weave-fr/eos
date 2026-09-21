"""Text tokenization pipeline for EOS."""

from __future__ import annotations

from pathlib import Path

from eos.tokenizer import EOSTokenizer


def tokenize_text(
    text: str,
    tokenizer: EOSTokenizer,
) -> list[int]:
    """Tokenize a text string."""

    if not text.strip():
        return []

    return tokenizer.encode(text)


def tokenize_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: EOSTokenizer,
) -> int:
    """Tokenize a UTF-8 text file and save token IDs."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    text = input_path.read_text(
        encoding="utf-8",
    )

    token_ids = tokenize_text(
        text,
        tokenizer,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        " ".join(str(token_id) for token_id in token_ids),
        encoding="utf-8",
    )

    return len(token_ids)
