# SAG2

This repository contains the SAG2 benchmark implementation, MuSiQue data used in the local experiments, and the experiment notes prepared during this study.

The retained upstream benchmark code and result records are under `SAG-Benchmark/`. The notes and implementation plans are under `doc/`.

## Repository layout

```text
dataset/
  musique.json
  musique_corpus.json

doc/
  SAG2_Experiment_Program.md
  SAG2_Hybrid_Experiment_Plan.md
  SAG2_Hybrid_Implementation_Plan.md

SAG-Benchmark/
  sag2/
    scripts/
    tests/
    prompts/
    results/
  sag/
```

## Current experiment scope

The planned study compares a BM25 initial retriever with a BM25 + BGE-M3 RRF hybrid retriever while keeping the Qwen evidence scorer, candidate budget, expansion rules, and evaluation protocol fixed.

The paper's hosted JEV results remain external reference values. A Qwen chat model used through an OpenAI-compatible endpoint is not the same scorer as the JEV model used in the paper.

## Environment

The local target environment is:

```text
Windows 11
Python 3.12
D:\learn\.venv
```

Install the `SAG-Benchmark/sag2` dependencies before running the benchmark:

```powershell
Set-Location .\SAG-Benchmark\sag2
uv sync --frozen
```

Run the offline tests:

```powershell
uv run pytest -q
uv run ruff check scripts tests
```

Run the offline paper-result checks:

```powershell
uv run python scripts/verify_paper_results.py
uv run python scripts/verify_agentic_results.py
uv run python scripts/verify_rerank_results.py
```

## Configuration

Copy the example configuration to a local file:

```powershell
Copy-Item .env.example .env
```

Fill in credentials only in the local `.env`. Never commit real API keys.

## Upstream documentation

- [English benchmark guide](SAG-Benchmark/README.md)
- [中文基准说明](SAG-Benchmark/README-CN.md)
- [SAG2 implementation](SAG-Benchmark/sag2/README.md)

## Data and result provenance

The benchmark implementation, prompts, and historical result records originate from the SAG-Benchmark project. MuSiQue questions and corpus passages are included to make the documented experiments directly runnable.

Check the nested license and upstream notices before redistributing or using third-party components.
