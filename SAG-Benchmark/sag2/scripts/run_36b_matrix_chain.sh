#!/usr/bin/env bash
# Full matrix under the chosen judge qwen3.6-27b.
#
#   retrieval ∈ {bm25, dense, hybrid}  ×  scoring ∈ {binary, grade}
#
# Arm layout (output/36b-100/H1/):
#
#   A1    BM25   + binary     <- reused from judge-100/A1-36-27B
#   B1    hybrid + binary
#   C1    dense  + binary
#   D1    BM25   + 6-band grade (τ=0.4, no-fill semantics via fill=0)
#   D1B   hybrid + 6-band grade
#   D1C   dense  + 6-band grade
#
# Each arm is audited; failure stops the chain.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/36b-100/H1
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.6-27b
ANCHOR=../../output/judge-100/H1/A1-36-27B

mkdir -p "$OUT"

# Reuse the finished BM25+binary arm instead of paying for it twice.
if [[ ! -e "$OUT/A1" ]]; then
  ln -s "$(realpath "$ANCHOR")" "$OUT/A1"
  echo "=== linked A1 -> $ANCHOR ==="
fi
"$PY" -m scripts.diagnostics.check_model_consistency "$OUT/A1" || {
  echo "!!! reused A1 failed audit" >&2
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

# --- binary row ---
run_arm B1 --retrieval-mode hybrid \
  --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale binary || exit 1

run_arm C1 --retrieval-mode dense \
  --scoring-scale binary || exit 1

# --- grade row ---
run_arm D1 --retrieval-mode bm25 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

run_arm D1B --retrieval-mode hybrid \
  --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

run_arm D1C --retrieval-mode dense \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 || exit 1

echo "=== $(date '+%F %T') 36b matrix chain complete ==="
echo
echo "=== paired vs A1 (BM25+binary @ qwen3.6-27b) ==="
for arm in B1 C1 D1 D1B D1C; do
  echo "----- $arm -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/A1" --arm "$OUT/$arm"
done
