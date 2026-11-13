#!/usr/bin/env bash
# Legacy 3.8 grade arms — kept for resume; new work uses run_38b_fill_matrix_chain.sh
# Naming: {A|B|C}{2|6}-H{n}-{model}  (see doc/SAG2_Arm_Naming_zh.md)
#
#   A6-H1-qwen3.8-27b   BM25   + grade
#   B6-H1-qwen3.8-27b   hybrid + grade
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/27b-100/H1
MODEL=qwen3.8-27b
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset

common=(
  --dataset-dir "$DATASET_DIR"
  --evidence-model "$MODEL"
  --candidate-k 100 --seed-k 10 --max-hops 1 --top-k 10
  --evidence-batch-size 10
  --concurrency 6 --evidence-concurrency 1
  --timeout 300
  --fill-missing-score 0
  --sample-file "$SAMPLE"
)

run_arm() {
  local name=$1
  shift
  if [[ -f "$OUT/$name/metrics.json" ]]; then
    echo "=== $(date '+%F %T') skip $name (already complete) ==="
    "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name" || exit 1
    return 0
  fi
  echo "=== $(date '+%F %T') starting $name ==="
  "$PY" scripts/run_hybrid_compare.py "${common[@]}" "$@" \
    --output-dir "$OUT/$name" 2>&1
  local rc=$?
  echo "=== $(date '+%F %T') $name exited rc=$rc ==="
  if [[ $rc -ne 0 ]]; then
    echo "!!! $name failed; stopping the chain" >&2
    exit 1
  fi
  "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name" || exit 1
}

run_arm A6-H1-qwen3.8-27b --retrieval-mode bm25 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

run_arm B6-H1-qwen3.8-27b --retrieval-mode hybrid \
  --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

echo "=== $(date '+%F %T') chain complete ==="
