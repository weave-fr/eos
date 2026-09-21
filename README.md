# EOS

**EOS — Intelligence begins.**

EOS is the first intelligence of Weave.

EOS is a general-purpose conversational AI model designed to help people understand, reason, create, and work with information.

## Vision

> **« EOS n'est pas simplement un modèle d'IA. C'est la première intelligence de Weave. »**

EOS is designed around a simple principle:

**The human provides the intention. EOS provides the intelligence. Together, they build the result.**

## EOS 0.1

EOS 0.1 is the first experimental and usable version of the EOS model family.

Its initial objective is simple:

> When someone opens EOS for the first time, they should be able to talk to it naturally, ask a question, request help, or create something, and receive a clear, natural, and useful response.

### Priorities

1. Usefulness
2. Simplicity
3. Reliability
4. Intelligence
5. Transparency
6. Safety
7. Adaptability
8. Speed
9. Creativity

## Core capabilities

EOS 0.1 is designed to support:

- French and English conversation
- Natural language understanding
- Multi-step reasoning
- General knowledge
- Explanations
- Writing and rewriting
- Summarization
- Brainstorming
- Planning
- Translation
- Creative generation
- Code generation
- Code analysis
- Bug fixing
- Document understanding
- Image and screenshot understanding
- Web search through external tools
- Calculator and other tools

EOS 0.1 will **not execute generated code**.

## Model philosophy

EOS should be:

- Natural
- Clear
- Useful
- Precise
- Adaptive
- Transparent
- Privacy-conscious
- Honest about uncertainty

EOS should never pretend to know something it does not know.

It should not become unnecessarily complex, expensive, opaque, or dependent on proprietary technologies that Weave cannot reasonably control.

## Technical direction

EOS 0.1 is designed around a modern dense decoder-only Transformer architecture.

Initial target:

- Approximately 200M parameters
- 8K context
- French-first multilingual capability
- French and English optimization
- Existing tokenizer
- RoPE positional encoding
- RMSNorm
- SwiGLU
- Grouped Query Attention
- Quantization-ready architecture
- Mixed-precision training

The 200M target is not a requirement if free compute cannot support it.

EOS development will progress through smaller models first to validate the complete pipeline before attempting the final EOS 0.1 target.

## Development strategy

EOS is being developed with a **€0 budget**.

The project therefore prioritizes:

- Free resources
- Open-source software
- Legally usable data
- Open and authorized datasets
- Efficient architectures
- Reproducible pipelines
- Checkpointing
- Progressive scaling

The goal is not to build the largest possible model.

The goal is to build the **smallest genuinely useful EOS**.

## Data

EOS uses a strict data pipeline.

Priority areas:

1. Conversation
2. Reasoning
3. Knowledge
4. Science
5. Mathematics
6. Technical documentation
7. Code
8. Creativity
9. Books and other texts

The dataset should prioritize quality over raw quantity.

Synthetic data may be used selectively when it provides a measurable improvement.

Human data should be minimized and handled with privacy in mind.

Potential data sources must be legally usable.

## Training

EOS development is planned in progressive stages:

1. Tiny prototype
2. Training pipeline validation
3. Small-scale pretraining
4. Larger-scale pretraining
5. Instruction tuning
6. Progressive reasoning training
7. Preference alignment
8. Evaluation
9. Private beta
10. Further iterations

EOS 0.1 does not rely on reinforcement learning.

Human preference data may be used for alignment without requiring RL.

## Reasoning

EOS is designed to support multi-step reasoning.

Detailed internal reasoning should remain hidden by default.

The model should instead provide the useful conclusion, explanation, or result needed by the user.

EOS should also perform internal self-checking where practical.

A dedicated reasoning mode may be introduced in a future EOS version.

## Conversation

EOS 0.1 targets approximately an 8K-token context window.

It should:

- Understand previous messages
- Maintain conversational context
- Adapt its response style
- Produce concise or detailed answers depending on the request

EOS 0.1 does not include persistent user memory.

## Tools

EOS is designed with a tool architecture from the beginning.

Initial tools may include:

- Web search
- Calculator
- Document processing
- Extensible external tools

More advanced multi-tool planning and autonomous agents are planned for later versions.

## Documents and vision

EOS will progressively support:

- TXT
- Markdown
- PDF
- JSON
- CSV
- Source code
- Images
- Screenshots

The first vision capabilities will focus on understanding existing images and documents rather than training a complete vision system from scratch.

## Safety

EOS should provide contextual and useful refusals when necessary.

The system will be tested against:

- Harmful requests
- Hallucinations
- Prompt injection
- Malicious documents
- Tool misuse
- Adversarial inputs
- Privacy risks

EOS should explicitly communicate uncertainty when appropriate.

## Privacy

EOS is designed around:

- Minimal personal data
- Opt-in use of conversations for improvement
- Data deletion
- Data export
- Isolated tools
- Privacy-conscious architecture

## Evaluation

Every EOS version should be evaluated before release.

The evaluation system will include:

- Internal benchmarks
- Human testing
- Capability tests
- Reasoning tests
- French-language tests
- English-language tests
- Safety tests
- Hallucination tests
- Prompt-injection tests
- Tool tests
- Malicious-document tests

A version should only be released when it reaches predefined quality thresholds.

## EOS versions

The planned evolution is:
```text
EOS 0.1
   ↓
EOS 0.2
   ↓
EOS 0.3
   ↓
EOS 1.0

Older versions may be retained temporarily when useful for comparison and development.

EOS family

Future models may include:

EOS

EOS Mini

EOS Pro

EOS Code

EOS Vision

EOS Reasoning


These models should share a common technical foundation where practical.

Weave architecture

EOS is the central intelligence layer of the future Weave ecosystem.

WEAVE
  │
  ▼
 EOS
  │
  ├── Chat
  ├── Code
  └── Agents
       │
       ▼
  Weave Builder
       │
       ▼
     Apps

EOS Code is intended to become the intelligence behind future Weave Builder workflows.

The long-term vision is:

Idea
  ↓
Conception
  ↓
UX/UI
  ↓
Prototype
  ↓
Code
  ↓
Tests
  ↓
Functional Web App

Roadmap

Phase 1 — Foundation

[x] Define EOS identity

[x] Define EOS 0.1 objectives

[x] Define technical direction

[ ] Create repository structure

[ ] Validate development environment

[ ] Select tokenizer

[ ] Build model prototype


Phase 2 — Training

[ ] Build data pipeline

[ ] Build dataset validation

[ ] Train tiny prototype

[ ] Validate loss and generation

[ ] Scale model progressively

[ ] Train EOS 0.1


Phase 3 — Intelligence

[ ] Instruction tuning

[ ] Reasoning training

[ ] Preference alignment

[ ] Self-checking

[ ] Evaluation suite

[ ] Safety testing


Phase 4 — EOS Interface

[ ] Inference system

[ ] Streaming generation

[ ] Web interface

[ ] Document support

[ ] Image support

[ ] Tool integration


Phase 5 — Beta

[ ] Private beta

[ ] Human testing

[ ] Feedback pipeline

[ ] EOS 0.2

[ ] Expanded beta


Phase 6 — Ecosystem

[ ] EOS API

[ ] EOS Mini

[ ] EOS Pro

[ ] EOS Code

[ ] Agents

[ ] Weave Builder


Project structure

eos/
├── README.md
├── pyproject.toml
├── configs/
│   └── eos_0_1.yaml
├── eos/
│   ├── model/
│   ├── tokenizer/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── scripts/
└── tests/

License

The licensing strategy has not yet been finalized.

Weave

EOS is developed as part of Weave.

Weave — Create with intelligence.

EOS — Intelligence begins.
