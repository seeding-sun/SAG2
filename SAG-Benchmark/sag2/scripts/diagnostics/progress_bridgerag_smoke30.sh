#!/usr/bin/env bash
OUT="$(cd "$(dirname "$0")" && pwd)"
echo "==== $(date -Is) ===="
pgrep -af 'run_bridgerag_smoke|bridgerag-compare-smoke30' || echo 'NO PROCESS'
for a in BridgeRAG-ours C6-MQ-H2-dual; do
  n=0
  [ -f "$OUT/$a/details.jsonl" ] && n=$(grep -c . "$OUT/$a/details.jsonl" || true)
  m='-'
  if [ -f "$OUT/$a/metrics.json" ]; then
    m=$(python3 -c "import json;d=json.load(open('$OUT/$a/metrics.json'));x=d.get('metrics') or {};print('R@5=%.4f R@10=%.4f'%(x.get('Recall@5',-1),x.get('Recall@10',-1)))")
  fi
  printf '%-18s %2s/30  %s\n' "$a" "$n" "$m"
done
