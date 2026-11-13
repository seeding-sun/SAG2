#!/usr/bin/env bash
# Fill the remaining qwen3.8-27b matrix cells (A/B/C × 2/6 × H0/H1/H2).
#
# Already complete (skipped):
#   A2-H1, B2-H1, C2-H1
#   A6-H0, A6-H1, A6-H2
#   B6-H1
#   C6-H0, C6-H1, C6-H2
#
# This chain adds:
#   A2-H0, A2-H2
#   B2-H0, B2-H2
#   C2-H0, C2-H2
#   B6-H0, B6-H2
#
# H0/H2 request caches are seeded from the matching H1 arm.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=../../output/27b-100/H1
SAMPLE=../../dataset/musique_100_for_ab.json
DATASET_DIR=../../dataset
MODEL=qwen3.8-27b

seed_requests_from_h1() {
  local name=$1
  local h1=$2
  mkdir -p "$OUT/$name"
  if [[ -d "$OUT/$h1/requests" && ! -d "$OUT/$name/requests" ]]; then
    echo "=== seeding request cache $h1 -> $name ==="
    cp -a "$OUT/$h1/requests" "$OUT/$name/requests"
  elif [[ -d "$OUT/$h1/requests" && -d "$OUT/$name/requests" ]]; then
    echo "=== merging request cache $h1 -> $name ==="
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
  local resume_flag=()
  [[ -d "$OUT/$name" ]] && resume_flag=(--resume)
  echo "=== $(date '+%F %T') starting $name ==="
  "$PY" scripts/run_hybrid_compare.py \
    --dataset-dir "$DATASET_DIR" \
    --evidence-model "$MODEL" \
    --candidate-k 100 --seed-k 10 --top-k 10 \
    --evidence-batch-size 10 \
    --concurrency 6 --evidence-concurrency 1 \
    --timeout 300 \
    --fill-missing-score 0 \
    --sample-file "$SAMPLE" \
    "${resume_flag[@]}" \
    "$@" \
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

# --- binary H0 (cache hits) ---
run_arm A2-H0-qwen3.8-27b A2-H1-qwen3.8-27b \
  --retrieval-mode bm25 --scoring-scale binary --max-hops 0 || exit 1

run_arm B2-H0-qwen3.8-27b B2-H1-qwen3.8-27b \
  --retrieval-mode hybrid --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale binary --max-hops 0 || exit 1

run_arm C2-H0-qwen3.8-27b C2-H1-qwen3.8-27b \
  --retrieval-mode dense --scoring-scale binary --max-hops 0 || exit 1

# --- grade H0 for Hybrid ---
run_arm B6-H0-qwen3.8-27b B6-H1-qwen3.8-27b \
  --retrieval-mode hybrid --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 --max-hops 0 || exit 1

# --- binary H2 ---
run_arm A2-H2-qwen3.8-27b A2-H1-qwen3.8-27b \
  --retrieval-mode bm25 --scoring-scale binary --max-hops 2 || exit 1

run_arm B2-H2-qwen3.8-27b B2-H1-qwen3.8-27b \
  --retrieval-mode hybrid --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale binary --max-hops 2 || exit 1

run_arm C2-H2-qwen3.8-27b C2-H1-qwen3.8-27b \
  --retrieval-mode dense --scoring-scale binary --max-hops 2 || exit 1

# --- grade H2 for Hybrid ---
run_arm B6-H2-qwen3.8-27b B6-H1-qwen3.8-27b \
  --retrieval-mode hybrid --rrf-k 60 --pool-k 100 --bm25-weight 1.0 --dense-weight 1.0 \
  --scoring-scale grade --grade-threshold 0.4 \
  --max-output-tokens-per-candidate 32 --max-hops 2 || exit 1

echo "=== $(date '+%F %T') 3.8 matrix fill-rest chain complete ==="
echo
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("../../output/27b-100/H1")
print(f"{'':4s}" + "".join(f"{'H'+str(h):>10s}" for h in (0,1,2)))
for letter, ret in (("A","bm25"),("B","hybrid"),("C","dense")):
  for scale in ("2","6"):
    row = f"{letter}{scale} "
    for h in (0,1,2):
      p = root / f"{letter}{scale}-H{h}-qwen3.8-27b" / "metrics.json"
      if p.exists():
        r = json.load(open(p))["metrics"]["Recall@10"]
        row += f"{r:10.4f}"
      else:
        row += f"{'—':>10s}"
    print(row)
PY
