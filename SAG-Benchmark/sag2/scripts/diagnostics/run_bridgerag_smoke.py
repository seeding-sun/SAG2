#!/usr/bin/env python3
"""BridgeRAG-style pipeline with our bge-m3 + qwen3.8-27b (matched-compare arm A).

Stages (paper-faithful structure, local models):
  1) hop1 ANN → bridge b = top-1
  2) LLM: extract e1/e2 from b; generate 3 SVO queries from (q,b)
  3) ANN: 3×SVO + e1 + e2 → union pool (≤20 unique full_docs)
  4) LLM tripartite judge s(q,b,c) over pool → rank → R@5/R@10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from embedding_client import EmbeddingClient  # noqa: E402
from qwen_evidence_scorer import chat_completions_endpoint  # noqa: E402
from run_hybrid_compare import load_subset  # noqa: E402
from run_sag2_benchmark import load_dataset, unique_documents, write_json  # noqa: E402
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, corpus_fingerprint  # noqa: E402
from sample_identity import index_unique_records  # noqa: E402
from vector_retriever import VectorCache, VectorRetriever  # noqa: E402


def recall_at(ranked_docs: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(ranked_docs[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def clip(text: str, n: int = 1200) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "…"


async def chat_json(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict | list:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    async with session.post(endpoint, headers=headers, json=payload) as resp:
        body = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"chat HTTP {resp.status}: {body[:400]}")
        data = json.loads(body)
    content = data["choices"][0]["message"]["content"]
    # tolerate fenced json
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


async def extract_entities(session, **kw) -> tuple[str, str]:
    system = (
        "Extract two short entity strings from the bridge passage for multi-hop QA. "
        "Return JSON: {\"e1\": bridge intermediate entity, \"e2\": hop-2 target entity inferred from the question}. "
        "Use empty string if unknown."
    )
    user = f"Question:\n{kw['question']}\n\nBridge passage:\n{clip(kw['bridge'])}\n"
    obj = await chat_json(
        session,
        endpoint=kw["endpoint"],
        api_key=kw["api_key"],
        model=kw["model"],
        system=system,
        user=user,
        max_tokens=128,
    )
    return str(obj.get("e1") or "").strip(), str(obj.get("e2") or "").strip()


async def generate_svo(session, **kw) -> list[str]:
    system = (
        "Generate exactly 3 hop-2 retrieval queries in short Subject-Verb-Object style, "
        "conditioned on the question and bridge passage. "
        'Return JSON: {"queries": ["...", "...", "..."]}.'
    )
    user = f"Question:\n{kw['question']}\n\nBridge passage:\n{clip(kw['bridge'])}\n"
    obj = await chat_json(
        session,
        endpoint=kw["endpoint"],
        api_key=kw["api_key"],
        model=kw["model"],
        system=system,
        user=user,
        max_tokens=256,
    )
    qs = obj.get("queries") or []
    out = [str(q).strip() for q in qs if str(q).strip()]
    return out[:3]


async def tripartite_scores(
    session,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    question: str,
    bridge: str,
    e1: str,
    e2: str,
    candidates: list[dict],
    batch_size: int = 8,
) -> dict[int, float]:
    """Return chunk_id -> score in [0,1]."""
    scores: dict[int, float] = {}
    system = (
        "You are a bridge-conditioned multi-hop evidence judge. "
        "Given question q, bridge passage b (hop-1), entities e1/e2, and a candidate passage c, "
        "score how useful c is as the NEXT hop to answer q given b. "
        "Score 0.0–1.0. Return JSON: {\"scores\": [{\"id\": <int>, \"score\": <float>}, ...]} "
        "with one entry per candidate id."
    )
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        lines = []
        for c in batch:
            lines.append(
                f"[id={c['id']}]\ntitle: {c.get('title') or ''}\n"
                f"text: {clip(c.get('content') or c.get('full_doc') or '', 800)}"
            )
        user = (
            f"Question q:\n{question}\n\n"
            f"Bridge b:\n{clip(bridge, 1000)}\n\n"
            f"e1: {e1}\ne2: {e2}\n\n"
            f"Candidates:\n" + "\n\n".join(lines) + "\n"
        )
        obj = await chat_json(
            session,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            max_tokens=256,
        )
        for item in obj.get("scores") or []:
            try:
                cid = int(item["id"])
                scores[cid] = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
        # fill missing with 0
        for c in batch:
            scores.setdefault(int(c["id"]), 0.0)
    return scores


def merge_hits(hit_lists: list[list[dict]], *, pool_n: int) -> list[dict]:
    best: dict[str, dict] = {}
    for hits in hit_lists:
        for h in hits:
            doc = h.get("full_doc") or ""
            if not doc:
                continue
            prev = best.get(doc)
            sc = float(h.get("dense_score") or h.get("score") or 0.0)
            if prev is None or sc > float(prev.get("dense_score") or 0.0):
                best[doc] = h
    ranked = sorted(
        best.values(),
        key=lambda h: (-float(h.get("dense_score") or 0.0), int(h["id"])),
    )
    return ranked[:pool_n]


async def run_one(
    *,
    session,
    index,
    vector: VectorRetriever,
    record: dict,
    endpoint: str,
    api_key: str,
    model: str,
    hop1_k: int,
    svo_k: int,
    entity_k: int,
    pool_n: int,
    top_k: int,
) -> dict:
    q = record["question"]
    gold = record["gold_docs"]
    hop1, _ = await vector.search(index, q, limit=hop1_k)
    if not hop1:
        empty = [""] * top_k
        return {
            "dataset_sample_id": record["id"],
            "question": q,
            "bridge_title": None,
            "e1": "",
            "e2": "",
            "svo_queries": [],
            "pool_size": 0,
            "retrieved_docs": empty,
            "metrics": {f"Recall@{k}": 0.0 for k in (1, 2, 5, 10)},
        }
    bridge = hop1[0]
    bridge_text = bridge.get("full_doc") or bridge.get("content") or ""
    e1, e2 = await extract_entities(
        session,
        question=q,
        bridge=bridge_text,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
    )
    svo = await generate_svo(
        session,
        question=q,
        bridge=bridge_text,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
    )
    lists: list[list[dict]] = [hop1]
    for sq in svo:
        hits, _ = await vector.search(index, sq, limit=svo_k)
        lists.append(hits)
    for ent in (e1, e2):
        if ent:
            hits, _ = await vector.search(index, ent, limit=entity_k)
            lists.append(hits)
    pool = merge_hits(lists, pool_n=pool_n)
    # ensure bridge present
    bdoc = bridge.get("full_doc")
    if bdoc and all(c.get("full_doc") != bdoc for c in pool):
        pool = [bridge, *pool][:pool_n]

    score_map = await tripartite_scores(
        session,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        question=q,
        bridge=bridge_text,
        e1=e1,
        e2=e2,
        candidates=pool,
    )
    ranked = sorted(
        pool,
        key=lambda c: (-score_map.get(int(c["id"]), 0.0), -float(c.get("dense_score") or 0), int(c["id"])),
    )
    # unique docs for metrics
    docs = []
    seen = set()
    for c in ranked:
        d = c.get("full_doc") or ""
        if d and d not in seen:
            seen.add(d)
            docs.append(d)
    metrics = {f"Recall@{k}": recall_at(docs, gold, k) for k in (1, 2, 5, 10)}
    return {
        "dataset_sample_id": record["id"],
        "question": q,
        "bridge_title": bridge.get("title"),
        "bridge_id": bridge.get("id"),
        "e1": e1,
        "e2": e2,
        "svo_queries": svo,
        "pool_size": len(pool),
        "judge_scores": {str(k): v for k, v in score_map.items()},
        "retrieved_docs": docs[:top_k],
        "metrics": metrics,
    }


async def run(args: argparse.Namespace) -> None:
    corpus, records, unit = load_dataset("musique", args.dataset_dir)
    selected = load_subset(args.sample_file, records)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    corpus_sha = corpus_fingerprint(args.dataset_dir / "musique_corpus.json")
    index = CorpusIndex(
        args.cache_dir / "musique.sqlite", corpus, corpus_sha, DEFAULT_CHUNK_CHARS
    )
    vector_cache = VectorCache.load(args.cache_dir, "musique")
    vector_cache.validate_index(index, corpus_sha)

    config = {
        "arm": "BridgeRAG-ours-models",
        "embedding": os.environ.get("SAG_EMBEDDING_MODEL"),
        "llm": args.model,
        "sample_file": str(args.sample_file),
        "sample_count": len(selected),
        "hop1_k": args.hop1_k,
        "svo_k": args.svo_k,
        "entity_k": args.entity_k,
        "pool_n": args.pool_n,
        "metric_unit": unit,
    }
    write_json(out / "config.json", config)

    details = out / "details.jsonl"
    completed = []
    if details.exists() and args.resume:
        completed = [
            json.loads(l)
            for l in details.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    done = index_unique_records(completed, "dataset_sample_id", "ckpt") if completed else {}
    pending = [r for r in selected if r["id"] not in done]
    print(f"pending {len(pending)}/{len(selected)}")

    endpoint = chat_completions_endpoint(os.environ["SAG_EVIDENCE_BASE_URL"])
    api_key = os.environ["SAG_EVIDENCE_API_KEY"]
    emb_base = os.environ["SAG_EMBEDDING_BASE_URL"]
    emb_model = os.environ["SAG_EMBEDDING_MODEL"]
    emb_key = os.environ.get("SAG_EMBEDDING_API_KEY", "")

    started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        emb = EmbeddingClient(
            session,
            emb_base,
            emb_model,
            emb_key,
            journal_dir=args.cache_dir / "request-cache" / "query-embedding",
            batch_size=32,
            concurrency=2,
            normalize=True,
            expected_dimension=1024,
        )
        vector = VectorRetriever(emb, vector_cache)
        with details.open("a", encoding="utf-8") as fh:
            for i, record in enumerate(pending, start=1):
                row = await run_one(
                    session=session,
                    index=index,
                    vector=vector,
                    record=record,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=args.model,
                    hop1_k=args.hop1_k,
                    svo_k=args.svo_k,
                    entity_k=args.entity_k,
                    pool_n=args.pool_n,
                    top_k=args.top_k,
                )
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(
                    f"[{i}/{len(pending)}] {record['id']} "
                    f"R@5={row['metrics']['Recall@5']:.2f} R@10={row['metrics']['Recall@10']:.2f} "
                    f"pool={row['pool_size']}"
                )

    rows = [
        json.loads(l)
        for l in details.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    metrics = {
        k: sum(r["metrics"][k] for r in rows) / len(rows)
        for k in ("Recall@1", "Recall@2", "Recall@5", "Recall@10")
    }
    summary = {
        "status": "complete",
        "question_count": len(rows),
        "config": config,
        "metrics": metrics,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(out / "metrics.json", summary)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[4]
    sag2 = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--sample-file",
        type=Path,
        default=repo / "dataset" / "musique_30_bridgerag_smoke.json",
    )
    p.add_argument("--dataset-dir", type=Path, default=repo / "dataset")
    p.add_argument("--cache-dir", type=Path, default=sag2 / ".cache" / "sag2")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "output" / "bridgerag-compare-smoke30" / "BridgeRAG-ours",
    )
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--hop1-k", type=int, default=5)
    p.add_argument("--svo-k", type=int, default=10)
    p.add_argument("--entity-k", type=int, default=5)
    p.add_argument("--pool-n", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
