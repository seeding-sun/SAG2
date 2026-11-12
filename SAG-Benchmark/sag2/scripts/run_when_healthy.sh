#!/usr/bin/env bash
# Wait for the Qwen server to drain, then run one SAG2 arm alone.
# Usage: run_when_healthy.sh <output-dir> [extra args...]
set -u
cd /data/machine_learning/cmh/知识库/SAG2/SAG-Benchmark/sag2 || exit 1
set -a; . ./.env; set +a

OUT="$1"; shift

healthy() {
  .venv/bin/python -u - "$@" <<'EOF' >/dev/null 2>&1
import aiohttp, asyncio, os, sys, time
async def main():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as s:
            async with s.post(
                f"{os.environ['SAG_EVIDENCE_BASE_URL'].rstrip('/')}/chat/completions",
                json={"model": os.environ["SAG_EVIDENCE_MODEL"],
                      "messages": [{"role": "user", "content": "reply with the word ok"}],
                      "temperature": 0},
                headers={"Authorization": f"Bearer {os.environ['SAG_EVIDENCE_API_KEY']}"},
            ) as r:
                if r.status != 200:
                    sys.exit(1)
    except Exception:
        sys.exit(1)
asyncio.run(main())
EOF
}

for attempt in $(seq 1 240); do
  if healthy; then
    echo "server healthy after $((attempt * 60))s"
    break
  fi
  echo "probe $attempt: still blocked, waiting 60s"
  sleep 60
done

if ! healthy; then
  echo "server never recovered; giving up"
  exit 2
fi

echo "launching arm into $OLDPWD/$OUT"
exec .venv/bin/python scripts/run_hybrid_compare.py \
  --retrieval-mode bm25 --dataset musique --dataset-dir ../../dataset \
  --sample-file ../../dataset/musique_100_for_ab.json \
  --output-dir "$OUT" --resume \
  --max-hops 1 --candidate-k 100 --seed-k 10 --top-k 10 \
  --scoring-scale grade --grade-threshold 0.4 \
  --cache-dir .cache/sag2 "$@"
