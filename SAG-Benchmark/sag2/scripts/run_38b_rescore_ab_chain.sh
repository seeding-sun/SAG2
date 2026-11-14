#!/usr/bin/env bash
# Minimal 100Q rerun after history-full re-score (commit 845e354).
#
# Arms: A6-H1/H2, A2-H1/H2  (BM25 × grade/binary × 1–2 hops)
# Output isolated from the pre-rescore matrix:
#   output/27b-100/H1-rescore/
#
# Do NOT seed request caches from output/27b-100/H1/ — hop≥1 payloads changed.
# H2 arms may seed from their matching H1 arm *inside* H1-rescore/ (hop0/hop1 only).
#
# Naming: {A}{2|6}-H{n}-qwen3.8-27b
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/27b-100/H1-rescore
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.8-27b
LOG="$OUT/chain.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

# Prefer local .env (gateway URL / key); model is overridden explicitly below.
if [[ -f ./.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export SAG_EVIDENCE_MODEL="$MODEL"

common=(
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

seed_requests_from() {
  local name=$1
  local src=$2
  mkdir -p "$OUT/$name"
  if [[ -d "$OUT/$src/requests" && ! -d "$OUT/$name/requests" ]]; then
    echo "=== seeding request cache $src -> $name (within H1-rescore only) ==="
    cp -a "$OUT/$src/requests" "$OUT/$name/requests"
  elif [[ -d "$OUT/$src/requests" && -d "$OUT/$name/requests" ]]; then
    echo "=== merging request cache $src -> $name ==="
    cp -an "$OUT/$src/requests/." "$OUT/$name/requests/" 2>/dev/null || \
      rsync -a --ignore-existing "$OUT/$src/requests/" "$OUT/$name/requests/"
  fi
}

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

echo "=== $(date '+%F %T') rescore AB chain start  OUT=$OUT  MODEL=$MODEL ==="

run_arm A6-H1-qwen3.8-27b --retrieval-mode bm25 --max-hops 1 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

run_arm A2-H1-qwen3.8-27b --retrieval-mode bm25 --max-hops 1 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

seed_requests_from A6-H2-qwen3.8-27b A6-H1-qwen3.8-27b
run_arm A6-H2-qwen3.8-27b --retrieval-mode bm25 --max-hops 2 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

seed_requests_from A2-H2-qwen3.8-27b A2-H1-qwen3.8-27b
run_arm A2-H2-qwen3.8-27b --retrieval-mode bm25 --max-hops 2 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

echo "=== $(date '+%F %T') rescore AB chain complete ==="
echo
echo "=== pairwise vs A2-H1 (same rescore code) ==="
for arm in A6-H1-qwen3.8-27b A2-H2-qwen3.8-27b A6-H2-qwen3.8-27b; do
  echo "----- $arm -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/A2-H1-qwen3.8-27b" --arm "$OUT/$arm" || true
done
