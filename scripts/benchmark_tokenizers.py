"""Compare existing tokenizers for EOS."""

from __future__ import annotations

from dataclasses import dataclass

from eos.tokenizer import EOSTokenizer


@dataclass
class TokenizerResult:
    """Benchmark result for one tokenizer."""

    name: str
    token_count: int
    characters_per_token: float
    vocab_size: int


SAMPLES = [
    (
        "fr",
        "EOS est une intelligence artificielle développée "
        "par Weave. Elle doit comprendre le français "
        "naturellement et produire des réponses utiles.",
    ),
    (
        "en",
        "EOS is an artificial intelligence model developed "
        "by Weave. It should understand English naturally "
        "and generate useful responses.",
    ),
    (
        "code",
        """def hello(name: str) -> str:
    return f"Hello, {name}!"
""",
    ),
]


def benchmark_tokenizer(
    name: str,
) -> TokenizerResult:
    """Benchmark one tokenizer."""

    tokenizer = EOSTokenizer.from_pretrained(name)

    text = "\n".join(
        sample
        for _, sample in SAMPLES
    )

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    token_count = len(token_ids)
    character_count = len(text)

    characters_per_token = (
        character_count / token_count
        if token_count
        else 0.0
    )

    return TokenizerResult(
        name=name,
        token_count=token_count,
        characters_per_token=characters_per_token,
        vocab_size=tokenizer.vocab_size,
    )


def main() -> None:
    tokenizer_names = [
        "HuggingFaceTB/SmolLM3-3B",
        "Qwen/Qwen2.5-0.5B",
    ]

    print("=== EOS tokenizer benchmark ===")
    print()

    for name in tokenizer_names:
        print(f"Testing: {name}")

        result = benchmark_tokenizer(name)

        print(f"Vocabulary: {result.vocab_size:,}")
        print(f"Tokens: {result.token_count:,}")
        print(
            "Characters/token: "
            f"{result.characters_per_token:.2f}"
        )
        print()


if __name__ == "__main__":
    main()
