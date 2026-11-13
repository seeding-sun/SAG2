#!/usr/bin/env bash
# BM25 vs Dense under six-band scoring, hop curve H0/H1/H2 @ qwen3.8-27b.
#
# Already complete:
#   A6-H1-qwen3.8-27b
#   C6-H1-qwen3.8-27b
#
# This chain adds H0 and H2 for both retrievals. Hop-0 (and for H2 also hop-1)
# request caches are copied from the matching H1 arm so only new hops hit the
# gateway — same trick used for the original 122B H0→H1 continuation.
#
# Naming: {A|C}6-H{n}-qwen3.8-27b  (see doc/SAG2_Arm_Naming_zh.md)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/27b-100/H1
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.8-27b

common=(
  --dataset-dir "$DATASET_DIR"
  --evidence-model "$MODEL"
  --candidate-k 100 --seed-k 10 --top-k 10
  --evidence-batch-size 10
  --concurrency 6 --evidence-concurrency 1
  --timeout 300
  --fill-missing-score 0
  --scoring-scale grade --grade-threshold 0.4
  --max-output-tokens-per-candidate 32
  --sample-file "$SAMPLE"
)

seed_requests_from_h1() {
  local name=$1
  local h1=$2
  mkdir -p "$OUT/$name"
  if [[ -d "$OUT/$h1/requests" && ! -d "$OUT/$name/requests" ]]; then
    echo "=== seeding request cache $h1 -> $name ==="
    cp -a "$OUT/$h1/requests" "$OUT/$name/requests"
  elif [[ -d "$OUT/$h1/requests" && -d "$OUT/$name/requests" ]]; then
    # Resume case: merge any missing digests without overwriting newer attempts.
    echo "=== merging request cache $h1 -> $name ==="
    cp -an "$OUT/$h1/requests/." "$OUT/$name/requests/" 2>/dev/null || \
      rsync -a --ignore-existing "$OUT/$h1/requests/" "$OUT/$name/requests/"
  fi
}

run_arm() {
  local name=$1
  local h1=$2
  shift 2
  if [[ -f "$OUT/$name/metrics.json" ]]; then
    echo "=== $(date '+%F %T') skip $name (already complete) ==="
    "$PY" -m scripts.diagnostics.check_model_consistency "$OUT/$name" || exit 1
    return 0
  fi
  seed_requests_from_h1 "$name" "$h1"
  # Pre-seeding creates the output dir, so --resume is required even on a
  # first full run (details.jsonl is still empty / absent).
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

# Fast: H0 should be almost entirely cache hits from H1 hop-0.
run_arm A6-H0-qwen3.8-27b A6-H1-qwen3.8-27b \
  --retrieval-mode bm25 --max-hops 0 || exit 1

run_arm C6-H0-qwen3.8-27b C6-H1-qwen3.8-27b \
  --retrieval-mode dense --max-hops 0 || exit 1

# Slower: H2 reuses hop-0/1 from H1; only hop-2 is new.
run_arm A6-H2-qwen3.8-27b A6-H1-qwen3.8-27b \
  --retrieval-mode bm25 --max-hops 2 || exit 1

run_arm C6-H2-qwen3.8-27b C6-H1-qwen3.8-27b \
  --retrieval-mode dense --max-hops 2 || exit 1

echo "=== $(date '+%F %T') hop-curve chain complete ==="
echo
echo "=== A6 hop curve ==="
for arm in A6-H0-qwen3.8-27b A6-H1-qwen3.8-27b A6-H2-qwen3.8-27b; do
  "$PY" -c "import json;m=json.load(open('$OUT/$arm/metrics.json'));print('$arm', m['metrics']['Recall@10'])"
done
echo "=== C6 hop curve ==="
for arm in C6-H0-qwen3.8-27b C6-H1-qwen3.8-27b C6-H2-qwen3.8-27b; do
  "$PY" -c "import json;m=json.load(open('$OUT/$arm/metrics.json'));print('$arm', m['metrics']['Recall@10'])"
done
echo "=== paired A6 vs C6 at each H ==="
for h in 0 1 2; do
  echo "----- H$h -----"
  "$PY" scripts/diagnostics/compare_arms.py \
    --baseline "$OUT/A6-H${h}-qwen3.8-27b" \
    --arm "$OUT/C6-H${h}-qwen3.8-27b"
done
