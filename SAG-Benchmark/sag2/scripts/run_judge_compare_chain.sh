#!/usr/bin/env bash
# Isolate the evidence-judge effect under a fixed A1 recipe.
#
# Everything stays locked to the A1-27B configuration except --evidence-model:
#
#   A1-27B      qwen3.8-27b     (already finished; used as the anchor)
#   A1-36-27B   qwen3.6-27b
#   A1-gemma    gemma4-26b-a4b
#
# Only models that returned themselves in the connectivity probe are included.
# Each arm is audited with check_model_consistency; a silent substitution
# aborts the chain.
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
  # Pure key omissions used to abort the whole arm under binary (fill=None).
  # A1-27B finished with 0 imputations, so fill=0 is behaviourally matched when
  # the model obeys the protocol and only differs on the rare dropped-key path.
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

run_arm A1-36-27B qwen3.6-27b || exit 1
run_arm A1-gemma  gemma4-26b-a4b || exit 1

echo "=== $(date '+%F %T') judge-compare chain complete ==="
echo
echo "=== paired against A1-27B (qwen3.8-27b) ==="
"$PY" scripts/diagnostics/compare_arms.py \
  --baseline ../../output/27b-100/H1/A1-27B \
  --arm "$OUT/A1-36-27B"
"$PY" scripts/diagnostics/compare_arms.py \
  --baseline ../../output/27b-100/H1/A1-27B \
  --arm "$OUT/A1-gemma"
"$PY" scripts/diagnostics/compare_arms.py \
  --baseline "$OUT/A1-36-27B" \
  --arm "$OUT/A1-gemma"
