#!/usr/bin/env bash
# Full matrix under qwen3.6-27b.
# Naming: {A|B|C}{2|6}-H{n}-{model}  (see doc/SAG2_Arm_Naming_zh.md)
#
#   A2-H1-qwen3.6-27b   BM25   + binary   <- reused from judge-100
#   B2-H1-qwen3.6-27b   hybrid + binary
#   C2-H1-qwen3.6-27b   dense  + binary
#   A6-H1-qwen3.6-27b   BM25   + grade
#   B6-H1-qwen3.6-27b   hybrid + grade
#   C6-H1-qwen3.6-27b   dense  + grade
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/36b-100/H1
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.6-27b
ANCHOR=../../output/judge-100/H1/A2-H1-qwen3.6-27b

mkdir -p "$OUT"

if [[ ! -e "$OUT/A2-H1-qwen3.6-27b" ]]; then
  ln -s "$(realpath "$ANCHOR")" "$OUT/A2-H1-qwen3.6-27b"
  echo "=== linked A2-H1-qwen3.6-27b -> $ANCHOR ==="
fi
"$PY" -m scripts.diagnostics.check_model_consistency "$OUT/A2-H1-qwen3.6-27b" || {
  echo "!!! reused A2 failed audit" >&2
  exit 1
}

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
  "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name" || {
    echo "!!! $name is contaminated; stopping the chain" >&2
    exit 1
  }
}

run_arm B2-H1-qwen3.6-27b --retrieval-mode hybrid \
  --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale binary || exit 1

run_arm C2-H1-qwen3.6-27b --retrieval-mode dense \
  --scoring-scale binary || exit 1

run_arm A6-H1-qwen3.6-27b --retrieval-mode bm25 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

run_arm B6-H1-qwen3.6-27b --retrieval-mode hybrid \
  --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

run_arm C6-H1-qwen3.6-27b --retrieval-mode dense \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

echo "=== $(date '+%F %T') 3.6 matrix chain complete ==="
BASE=A2-H1-qwen3.6-27b
echo "=== paired vs $BASE ==="
for arm in B2-H1-qwen3.6-27b C2-H1-qwen3.6-27b A6-H1-qwen3.6-27b B6-H1-qwen3.6-27b C6-H1-qwen3.6-27b; do
  echo "----- $arm -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/$BASE" --arm "$OUT/$arm"
done
