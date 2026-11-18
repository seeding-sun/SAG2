#!/usr/bin/env bash
# 2WikiMultihopQA 100Q H1 minimal chain (rescore code).
# Arms: A6-H1, A2-H1, C6-H1  — see SAG2_CrossDataset_Experiment_Plan_zh.md
#
# Output: output/2wiki-100/H1-rescore/
# Do NOT seed from musique H1-rescore caches.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/2wiki-100/H1-rescore
SAMPLE=../../dataset/2wiki_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.8-27b
LOG="$OUT/chain.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

if [[ -f ./.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export SAG_EVIDENCE_MODEL="$MODEL"

common=(
  --dataset 2wikimultihopqa
  --dataset-dir "$DATASET_DIR"
  --evidence-model "$MODEL"
  --candidate-k 100 --seed-k 10 --top-k 10
  --evidence-batch-size 10
  --concurrency 6 --evidence-concurrency 1
  --timeout 300
  --fill-missing-score 0
  --max-output-tokens-per-candidate 32
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
  local resume_flag=()
  if [[ -d "$OUT/$name" ]]; then
    resume_flag=(--resume)
  fi
  echo "=== $(date '+%F %T') starting $name ==="
  "$PY" scripts/run_hybrid_compare.py "${common[@]}" "$@" \
    "${resume_flag[@]}" \
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

echo "=== $(date '+%F %T') 2wiki H1 chain start  OUT=$OUT  MODEL=$MODEL ==="
echo "sample=$SAMPLE"

run_arm A6-H1-qwen3.8-27b --retrieval-mode bm25 --max-hops 1 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

run_arm A2-H1-qwen3.8-27b --retrieval-mode bm25 --max-hops 1 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

run_arm C6-H1-qwen3.8-27b --retrieval-mode dense --max-hops 1 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

echo "=== $(date '+%F %T') 2wiki H1 chain complete ==="
echo
echo "=== pairwise vs A2-H1 ==="
for arm in A6-H1-qwen3.8-27b C6-H1-qwen3.8-27b; do
  echo "----- $arm -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/A2-H1-qwen3.8-27b" --arm "$OUT/$arm" || true
done
echo "=== pairwise C6 vs A6 ==="
"$PY" scripts/diagnostics/compare_arms.py \
  --baseline "$OUT/A6-H1-qwen3.8-27b" --arm "$OUT/C6-H1-qwen3.8-27b" || true
