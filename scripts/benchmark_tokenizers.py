"""Compare existing tokenizers for EOS."""

from __future__ import annotations

from dataclasses import dataclass

from eos.tokenizer import EOSTokenizer


@dataclass
class TokenizerResult:
    """Benchmark result for one tokenizer."""

    name: str
    vocab_size: int
    french_tokens: int
    english_tokens: int
    code_tokens: int
    total_tokens: int
    characters_per_token: float


SAMPLES = {
    "fr": """
EOS est une intelligence artificielle développée par Weave.
Elle doit comprendre les demandes en français, conserver le
contexte d'une conversation et produire des réponses claires,
naturelles et utiles.

Une intelligence artificielle doit également savoir reconnaître
l'incertitude, éviter d'inventer des informations et expliquer
simplement les concepts complexes.
""",
    "en": """
EOS is an artificial intelligence model developed by Weave.
It should understand English naturally, maintain conversational
context, reason about problems, and generate clear and useful
answers.

A language model should also recognize uncertainty and avoid
making up information when it does not know something.
""",
    "code": """
def calculate_average(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")

    return sum(values) / len(values)


class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def describe(self) -> str:
        return f"{self.name} is {self.age} years old."
""",
}


def benchmark_tokenizer(name: str) -> TokenizerResult:
    """Benchmark one tokenizer."""

    tokenizer = EOSTokenizer.from_pretrained(name)

    token_counts: dict[str, int] = {}

    total_characters = 0
    total_tokens = 0

    for category, text in SAMPLES.items():
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        count = len(token_ids)

        token_counts[category] = count
        total_tokens += count
        total_characters += len(text)

    characters_per_token = (
        total_characters / total_tokens
        if total_tokens
        else 0.0
    )

    return TokenizerResult(
        name=name,
        vocab_size=tokenizer.vocab_size,
        french_tokens=token_counts["fr"],
        english_tokens=token_counts["en"],
        code_tokens=token_counts["code"],
        total_tokens=total_tokens,
        characters_per_token=characters_per_token,
    )


def main() -> None:
    tokenizer_names = [
        "HuggingFaceTB/SmolLM3-3B",
        "Qwen/Qwen2.5-0.5B",
    ]

    print("=== EOS 0.1 TOKENIZER BENCHMARK ===")
    print()

    results: list[TokenizerResult] = []

    for name in tokenizer_names:
        print(f"Testing: {name}")

        result = benchmark_tokenizer(name)
        results.append(result)

        print(f"Vocabulary: {result.vocab_size:,}")
        print(f"French tokens: {result.french_tokens:,}")
        print(f"English tokens: {result.english_tokens:,}")
        print(f"Code tokens: {result.code_tokens:,}")
        print(f"Total tokens: {result.total_tokens:,}")
        print(
            "Characters/token: "
            f"{result.characters_per_token:.2f}"
        )
        print()

    print("=== SUMMARY ===")

    for result in results:
        print(
            f"{result.name}: "
            f"{result.total_tokens:,} tokens, "
            f"{result.vocab_size:,} vocabulary"
        )


if __name__ == "__main__":
    main()
