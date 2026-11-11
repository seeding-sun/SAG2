# Hybrid BM25 + BGE-M3 Experiment

This workflow compares:

```text
A: BM25 top-100            -> Qwen evidence scorer -> SAG2
B: BM25+BGE-M3 RRF top-100 -> Qwen evidence scorer -> SAG2
```

Only the initial retriever changes. Seed selection, term expansion, DPP ordering and evaluation stay the same.

## Requirements

Use Python 3.11 or newer and install the project:

```powershell
uv sync --frozen --dev
```

Configure model endpoints in the local `.env`:

```dotenv
SAG_EMBEDDING_BASE_URL=http://192.168.2.108:8013/v1
SAG_EMBEDDING_MODEL=bge-m3:latest
SAG_EMBEDDING_API_KEY=

SAG_EVIDENCE_BASE_URL=http://192.168.2.69:9099/v1
SAG_EVIDENCE_MODEL=qwen3.5-122b-a10b
SAG_EVIDENCE_API_KEY=your-local-key
```

Never commit the real `.env`.

## 1. Build the BM25 index

The example uses the MuSiQue files stored at the repository root:

```powershell
$env:PYTHONUTF8 = "1"
D:\learn\.venv\Scripts\python.exe scripts\run_sag2_benchmark.py `
  --datasets musique `
  --dataset-dir ..\..\dataset `
  --index-only `
  --index-workers 8
```

The SQLite index is written to `.cache/sag2/musique.sqlite`.

## 2. Create the fixed 100-question subset

```powershell
D:\learn\.venv\Scripts\python.exe scripts\make_musique_ab_subset.py `
  --dataset-dir ..\..\dataset `
  --output ..\..\dataset\musique_100_for_ab.json `
  --seed 20261111 `
  --strata 2hop:40,3hop:30,4hop:30
```

Both experiment arms must use the same sample file.

## 3. Build the BGE-M3 chunk embedding cache

```powershell
D:\learn\.venv\Scripts\python.exe scripts\build_passage_embeddings.py `
  --dataset-dir ..\..\dataset `
  --index .cache\sag2\musique.sqlite `
  --cache-dir .cache\sag2 `
  --batch-size 32 `
  --concurrency 2
```

The cache consists of:

```text
.cache/sag2/musique-bge-m3.npy
.cache/sag2/musique-bge-m3.ids.json
.cache/sag2/musique-bge-m3.manifest.json
```

The manifest binds the vectors to the corpus SHA256, index signature, chunk policy, model and dimension.

## 4. Evaluate initial retrieval without Qwen

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_initial_retrieval_eval.py `
  --dataset-dir ..\..\dataset `
  --sample-file ..\..\dataset\musique_100_for_ab.json `
  --cache-dir .cache\sag2 `
  --output-dir output\initial-retrieval
```

This compares BM25, dense retrieval and RRF fusion without making evidence-scorer requests. It reports top-k recall, complete-evidence coverage, full-entry recall and hybrid-added noise.

## 5. Run the A/B arms

BM25 arm:

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_hybrid_compare.py `
  --retrieval-mode bm25 `
  --dataset-dir ..\..\dataset `
  --sample-file ..\..\dataset\musique_100_for_ab.json `
  --output-dir output\hybrid-ab\A1-H1 `
  --max-hops 1 `
  --candidate-k 100 `
  --seed-k 10
```

Hybrid arm:

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_hybrid_compare.py `
  --retrieval-mode hybrid `
  --dataset-dir ..\..\dataset `
  --sample-file ..\..\dataset\musique_100_for_ab.json `
  --output-dir output\hybrid-ab\B1-H1 `
  --max-hops 1 `
  --candidate-k 100 `
  --seed-k 10
```

Dense-only arm:

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_hybrid_compare.py `
  --retrieval-mode dense `
  --dataset-dir ..\..\dataset `
  --sample-file ..\..\dataset\musique_100_for_ab.json `
  --output-dir output\hybrid-ab\C1-H1 `
  --max-hops 1 `
  --candidate-k 100 `
  --seed-k 10
```

Use `--max-hops 0` for the initial-scoring-only comparison.

Each output directory contains:

```text
config.json
details.jsonl
metrics.json
search_results.json
requests/
```

The runner is resumable:

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_hybrid_compare.py `
  --retrieval-mode bm25 `
  --dataset-dir ..\..\dataset `
  --sample-file ..\..\dataset\musique_100_for_ab.json `
  --output-dir output\hybrid-ab\A1-H1 `
  --max-hops 1 `
  --resume
```

## 6. Analyze paired results

```powershell
D:\learn\.venv\Scripts\python.exe scripts\analyze_hybrid_results.py `
  --a-dir output\hybrid-ab\A1-H1 `
  --b-dir output\hybrid-ab\B1-H1 `
  --output output\hybrid-ab\comparison.json
```

The analyzer writes paired metric deltas, 95% bootstrap intervals, hop-stratified changes, win/tie/loss counts, initial-candidate noise and cost totals.

## Notes

- `jev_score` keeps its historical field name so the existing SAG2 loop can be reused unchanged. A score produced by this adapter is a Qwen judgment, not an output from the paper's hosted JEV model.
- The BGE embedding dimension defaults to 1024. If the deployed model reports a different dimension, pass `--dimension` and `--embedding-dimension` consistently.
- The main comparison must not add a reranker to only one arm.
- The response cache is for resuming identical requests. Independent Qwen stability repeats must use separate output directories.
