#!/usr/bin/env bash
# Isolate the evidence-judge effect under a fixed A2-H1 recipe.
# Naming: {A|B|C}{2|6}-H{n}-{model}  (see doc/SAG2_Arm_Naming_zh.md)
#
#   A2-H1-qwen3.8-27b      (anchor under 27b-100)
#   A2-H1-qwen3.6-27b
#   A2-H1-gemma4-26b-a4b
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/judge-100/H1
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset

common=(
  --dataset-dir "$DATASET_DIR"
  --retrieval-mode bm25
  --scoring-scale binary
  --fill-missing-score 0
  --candidate-k 100 --seed-k 10 --max-hops 1 --top-k 10
  --evidence-batch-size 10
  --concurrency 6 --evidence-concurrency 1
  --timeout 300
  --sample-file "$SAMPLE"
)

run_arm() {
  local name=$1
  local model=$2
  if [[ -f "$OUT/$name/metrics.json" ]]; then
    echo "=== $(date '+%F %T') skip $name (already complete) ==="
    "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name" || exit 1
    return 0
  fi
  echo "=== $(date '+%F %T') starting $name (model=$model) ==="
  "$PY" scripts/run_hybrid_compare.py "${common[@]}" \
    --evidence-model "$model" \
    --output-dir "$OUT/$name" 2>&1
  local rc=$?
  echo "=== $(date '+%F %T') $name exited rc=$rc ==="
  if [[ $rc -ne 0 ]]; then
    echo "!!! $name failed; stopping the chain" >&2
    exit 1
  fi
  "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name"
  local audit=$?
  if [[ $audit -ne 0 ]]; then
    echo "!!! $name is contaminated; stopping the chain" >&2
    exit 1
  fi
}

mkdir -p "$OUT"

run_arm A2-H1-qwen3.6-27b qwen3.6-27b || exit 1
run_arm A2-H1-gemma4-26b-a4b gemma4-26b-a4b || exit 1

echo "=== $(date '+%F %T') judge-compare chain complete ==="
ANCHOR=../../output/27b-100/H1/A2-H1-qwen3.8-27b
"$PY" scripts/diagnostics/compare_arms.py --baseline "$ANCHOR" --arm "$OUT/A2-H1-qwen3.6-27b"
"$PY" scripts/diagnostics/compare_arms.py --baseline "$ANCHOR" --arm "$OUT/A2-H1-gemma4-26b-a4b"
"$PY" scripts/diagnostics/compare_arms.py --baseline "$OUT/A2-H1-qwen3.6-27b" --arm "$OUT/A2-H1-gemma4-26b-a4b"
