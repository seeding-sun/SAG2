# SAG2 Hybrid 10-Question Smoke Report

Date: 2026-11-11

## Scope

This smoke test compared two initial retrievers while keeping the SAG2 scoring and expansion pipeline fixed:

```text
A: BM25 initial top-100
B: BM25 + BGE-M3 RRF initial top-100
```

Both arms used:

```text
dataset             MuSiQue
questions           10 fixed questions
question mix        2-hop: 4, 3-hop: 3, 4-hop: 3
candidate_k         100
seed_k              10
top_k               10
embedding model     bge-m3:latest
embedding dimension 1024
evidence model      qwen3.5-122b-a10b
temperature         0
```

The service was configured with `qwen3.5-397b-a17b`, but the actual response model was `qwen3.5-122b-a10b`. Both arms used the actual returned model.

## Initial retrieval

The retrieval-only experiment used the fixed 100-question subset and did not call Qwen.

| Method | R@5 | R@10 | AE@10 | Full-entry recall |
|---|---:|---:|---:|---:|
| BM25 | 43.42% | 50.67% | 14.00% | 37.00% |
| BGE-M3 dense | 50.00% | 59.50% | 25.00% | 62.00% |
| BM25+BGE RRF | 52.67% | 58.00% | 22.00% | 55.00% |

Hybrid added 3,732 candidates relative to BM25 across 100 questions. Only 33 of the added candidates were gold evidence, giving a mean noise rate of 99.06%.

## H=0 smoke

| Metric | A: BM25+Qwen | B: Hybrid+Qwen | B - A |
|---|---:|---:|---:|
| R@1 | 30.00% | 30.00% | 0.00 pp |
| R@2 | 47.50% | 44.17% | -3.33 pp |
| R@5 | 60.00% | 49.17% | -10.83 pp |
| R@10 | 60.00% | 61.67% | +1.67 pp |
| AE@10 | 20.00% | 20.00% | 0.00 pp |
| Passage judgments | 1,000 | 1,000 | 0 |
| Input tokens | 158,252 | 174,129 | +15,877 |

Pairwise R@10:

```text
B wins: 2
Ties:   6
A wins: 2
```

The paired R@10 bootstrap interval was `[-10.83, +15.00]` percentage points.

## H=1 smoke

| Metric | A: BM25+Qwen | B: Hybrid+Qwen | B - A |
|---|---:|---:|---:|
| R@1 | 25.00% | 25.00% | 0.00 pp |
| R@2 | 30.83% | 41.67% | +10.83 pp |
| R@5 | 52.50% | 49.17% | -3.33 pp |
| R@10 | 57.50% | 49.17% | -8.33 pp |
| AE@10 | 20.00% | 10.00% | -10.00 pp |
| Passage judgments | 2,100 | 2,092 | -8 |

Pairwise R@10:

```text
B wins: 1
Ties:   6
A wins: 3
```

The paired R@10 bootstrap interval was `[-26.67, +10.00]` percentage points.

## Observations

1. BGE dense retrieval improves initial top-k coverage compared with BM25.
2. RRF improves initial R@5 relative to dense-only, but introduces substantial noise at top-100.
3. Under Qwen scoring at H=0, hybrid gains only 1.67 R@10 points and loses 10.83 R@5 points.
4. After one expansion, hybrid loses 8.33 R@10 points and 10 AE@10 points relative to BM25.
5. The 10-question intervals are wide, so these values are smoke-test diagnostics, not a final estimate.

## Decision

Do not run the full 100-question H=1 comparison unchanged. The current RRF construction appears to add too many noisy candidates for this Qwen scorer.

Recommended next steps:

1. Inspect no-entry, wrong-seed, no-bridge and wrong-bridge cases from both H=1 arms.
2. Compare per-question Qwen score distributions and seed choices for the hybrid-added candidates.
3. Test RRF balancing on a separate parameter-selection split before another confirmation run.
4. Keep the current 10-question samples only as regression fixtures; do not tune and report on the same questions.

## Artifacts

```text
D:\learn\知识库\RAG\SAG2\output\initial-retrieval-20261111\metrics.json
D:\learn\知识库\RAG\SAG2\output\hybrid-smoke\A1-H0\metrics.json
D:\learn\知识库\RAG\SAG2\output\hybrid-smoke\B1-H0\metrics.json
D:\learn\知识库\RAG\SAG2\output\hybrid-smoke\A1-H1\metrics.json
D:\learn\知识库\RAG\SAG2\output\hybrid-smoke\B1-H1\metrics.json
D:\learn\知识库\RAG\SAG2\output\hybrid-smoke\comparison-H0.md
D:\learn\知识库\RAG\SAG2\output\hybrid-smoke\comparison-H1.md
```
