#!/usr/bin/env bash
# 真正的 4 臂：各跑 max_hops=2；H0/H1 从同一次 run 的 hop_metrics 取，不重复打分。
set -euo pipefail
cd /data/machine_learning/cmh/知识库/SAG2/SAG-Benchmark/sag2
set -a; . ./.env; set +a
SAMPLE=../../dataset/musique_100_for_ab.json
OUT=../../output/mq-dual-100
PY=.venv/bin/python
LOG="$OUT/chain.log"

run_one () {
  local name="$1"; shift
  echo "==== START $name $(date -Is) ====" | tee -a "$LOG"
  $PY scripts/diagnostics/run_c6_mq_h1.py \
    --sample-file "$SAMPLE" \
    --output-dir "$OUT/$name" \
    --evidence-model qwen3.8-27b \
    --rewrite-model qwen3.8-27b \
    --max-hops 2 \
    --resume \
    "$@" \
    2>&1 | tee -a "$OUT/${name}.log"
  echo "==== DONE $name $(date -Is) ====" | tee -a "$LOG"
}

run_one C6-MQ-H2 --scoring-scale grade --grade-threshold 0.4
run_one C6-MQ-H2-dual --scoring-scale grade --grade-threshold 0.4 \
  --dual-anchor --dual-anchor-mode max --verify-retrieve --no-bridge-gate
run_one C2-MQ-H2 --scoring-scale binary --grade-threshold 0.5
run_one C2-MQ-H2-dual --scoring-scale binary --grade-threshold 0.5 \
  --dual-anchor --dual-anchor-mode max --verify-retrieve --no-bridge-gate

echo "ALL_DONE $(date -Is)" | tee -a "$LOG"
