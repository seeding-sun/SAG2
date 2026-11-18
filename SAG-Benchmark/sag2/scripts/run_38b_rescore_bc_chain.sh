#!/usr/bin/env bash
# Rescore-code 100Q: Hybrid/Dense × grade/binary × H1/H2.
#
# Continues after A6/A2 in output/27b-100/H1-rescore/ (same sag2_code_sha256).
# Do NOT seed from output/27b-100/H1/ — hop≥1 payloads changed.
# H2 may seed from matching H1 *inside* H1-rescore/ only.
#
# Naming: {B|C}{2|6}-H{n}-qwen3.8-27b
# Order: grade first (A6 won), then binary; H1 before H2; B then C within each hop.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/27b-100/H1-rescore
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.8-27b
LOG="$OUT/bc_chain.log"

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

hybrid=(--retrieval-mode hybrid --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0)
dense=(--retrieval-mode dense)

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

echo "=== $(date '+%F %T') rescore BC chain start  OUT=$OUT  MODEL=$MODEL ==="

# --- grade (align with winning A6) ---
run_arm B6-H1-qwen3.8-27b "${hybrid[@]}" --max-hops 1 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

run_arm C6-H1-qwen3.8-27b "${dense[@]}" --max-hops 1 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

seed_requests_from B6-H2-qwen3.8-27b B6-H1-qwen3.8-27b
run_arm B6-H2-qwen3.8-27b "${hybrid[@]}" --max-hops 2 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

seed_requests_from C6-H2-qwen3.8-27b C6-H1-qwen3.8-27b
run_arm C6-H2-qwen3.8-27b "${dense[@]}" --max-hops 2 \
  --scoring-scale grade --grade-threshold 0.4 || exit 1

# --- binary (align with A2; old matrix B2-H2 was pre-rescore best) ---
run_arm B2-H1-qwen3.8-27b "${hybrid[@]}" --max-hops 1 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

run_arm C2-H1-qwen3.8-27b "${dense[@]}" --max-hops 1 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

seed_requests_from B2-H2-qwen3.8-27b B2-H1-qwen3.8-27b
run_arm B2-H2-qwen3.8-27b "${hybrid[@]}" --max-hops 2 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

seed_requests_from C2-H2-qwen3.8-27b C2-H1-qwen3.8-27b
run_arm C2-H2-qwen3.8-27b "${dense[@]}" --max-hops 2 \
  --scoring-scale binary --grade-threshold 0.0 || exit 1

echo "=== $(date '+%F %T') rescore BC chain complete ==="
echo
echo "=== pairwise vs A6-H2 (rescore grade anchor) ==="
for arm in B6-H1-qwen3.8-27b C6-H1-qwen3.8-27b B6-H2-qwen3.8-27b C6-H2-qwen3.8-27b; do
  echo "----- $arm -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/A6-H2-qwen3.8-27b" --arm "$OUT/$arm" || true
done
echo "=== pairwise vs A2-H1 (rescore binary anchor) ==="
for arm in B2-H1-qwen3.8-27b C2-H1-qwen3.8-27b B2-H2-qwen3.8-27b C2-H2-qwen3.8-27b; do
  echo "----- $arm -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/A2-H1-qwen3.8-27b" --arm "$OUT/$arm" || true
done
