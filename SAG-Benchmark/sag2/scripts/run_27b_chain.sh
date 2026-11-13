#!/usr/bin/env bash
# Run the matched 27B arms in sequence.
#
# Switching the evidence model from qwen3.5-122b-a10b to qwen3.8-27b breaks
# comparability with the existing 122B runs, so every 27B claim needs its own
# baseline measured under the same model:
#
#   A1-27B     BM25   + binary            -> 27B baseline
#   D1-27B     BM25   + 6-band + no fill  -> does the D1 effect transfer?
#   D1B1-27B   hybrid + 6-band + no fill  -> target arm
#
# Each arm is preceded by the model-consistency audit, so a silent gateway
# substitution aborts the chain instead of producing an uninterpretable arm.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/27b-100/H1
MODEL=qwen3.8-27b
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
LOG=/tmp/27b-chain.log

common=(
  --dataset-dir "$DATASET_DIR"
  --evidence-model "$MODEL"
  --candidate-k 100 --seed-k 10 --max-hops 1 --top-k 10
  --evidence-batch-size 10
  --concurrency 6 --evidence-concurrency 1
  --timeout 300
  --sample-file "$SAMPLE"
)

run_arm() {
  local name=$1
  shift
  echo "=== $(date '+%F %T') starting $name ==="
  "$PY" scripts/run_hybrid_compare.py "${common[@]}" "$@" \
    --output-dir "$OUT/$name" 2>&1
  local rc=$?
  echo "=== $(date '+%F %T') $name exited rc=$rc ==="

  # The run itself now aborts on a model mismatch; audit anyway so cached
  # replays and older trees cannot slip through.
  "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name"
  local audit=$?
  if [[ $audit -ne 0 ]]; then
    echo "!!! $name is contaminated; stopping the chain" >&2
    exit 1
  fi
  return $rc
}

# Failures must stop the chain: a silently failed arm used to be reported as
# "chain complete" because the helper's exit status was never checked.
run_arm D1-27B --retrieval-mode bm25 \
  --scoring-scale grade --grade-threshold 0.4 --fill-missing-score 0 \
  --max-output-tokens-per-candidate 32 || exit 1

run_arm D1B1-27B --retrieval-mode hybrid \
  --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale grade --grade-threshold 0.4 --fill-missing-score 0 \
  --max-output-tokens-per-candidate 32 || exit 1

echo "=== $(date '+%F %T') chain complete ==="
