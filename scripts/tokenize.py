"""Tokenize a text file for EOS."""

from __future__ import annotations

import argparse

from eos.data import tokenize_file
from eos.tokenizer import EOSTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input UTF-8 text file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output token ID file.",
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer name or path.",
    )

    args = parser.parse_args()

    tokenizer = EOSTokenizer.from_pretrained(
        args.tokenizer,
    )

    token_count = tokenize_file(
        input_path=args.input,
        output_path=args.output,
        tokenizer=tokenizer,
    )

    print(f"Tokenized {token_count:,} tokens.")


if __name__ == "__main__":
    main()
