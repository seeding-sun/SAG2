#!/usr/bin/env python3
"""Probe: can query decompose + multi-path dense retrieval recover never-recalled golds?

Arms per question:
  - baseline: union of C6-H2 candidate document_ids across hops (expect 0 hits on never golds)
  - decomp_only: LLM sub-queries, each dense top-k, RRF-merge (no intermediate extract)
  - iterative: ordered sub-queries; after each retrieve, LLM extracts a short entity and
    fills the next sub-query before retrieving again

Usage (from sag2/):
  set -a; . ./.env; set +a
  .venv/bin/python scripts/diagnostics/probe_never_multiquery.py \\
    --never ../../output/27b-100/H1-rescore/diagnostics/never_C6_H2_fixed.json \\
    --run ../../output/27b-100/H1-rescore/C6-H2-qwen3.8-27b \\
    --out ../../output/probe-never-multiquery
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embedding_client import EmbeddingClient  # noqa: E402
from qwen_evidence_scorer import chat_completions_endpoint  # noqa: E402
from run_sag2_benchmark import load_dataset  # noqa: E402
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, compact_json, corpus_fingerprint  # noqa: E402
from vector_retriever import VectorCache, VectorRetriever  # noqa: E402


DECOMPOSE_SYSTEM = (
    "You decompose multi-hop questions into ordered sub-questions. "
    "Return only one JSON object."
)
DECOMPOSE_USER = (
    "Decompose the question into 3 to 5 ordered sub-questions that a retrieval system "
    "could answer step by step. Write each sub-question as a natural full question "
    "(Who/What/When/Where...), not 'X >> relation' shorthand. Use placeholders like "
    "{{1}}, {{2}} when a later sub-question depends on an earlier answer. Do not invent "
    "final entity names that are not implied by the question wording. "
    'Return {{"sub_questions": ["...", "..."]}} only.\n\n'
    "question: {question}"
)

EXTRACT_SYSTEM = (
    "You extract a short answer entity from retrieved passages. Return only one JSON object."
)
EXTRACT_USER = (
    "Given the sub-question and passages, extract the shortest answer string "
    "(entity name, place, date, or short phrase). Prefer explicit relational cues that "
    "match the sub-question (e.g. 'born in' for birthplace; 'headquartered in' for HQ; "
    "'produced by' for manufacturer). Do not treat album/marketing hometown blurbs "
    "(e.g. 'X, Ontario singer') as place-of-birth unless the passage explicitly says "
    "born/birthplace. If unknown, return an empty string. "
    'Return {{"answer": "..."}} only.\n\n'
    "sub_question: {sub_question}\n"
    "passages:\n{passages}"
)

FOLLOWUP_SYSTEM = (
    "You write short retrieval queries that chase missing multi-hop bridge entities. "
    "Return only one JSON object."
)
FOLLOWUP_USER = (
    "Original question:\n{question}\n\n"
    "Already retrieved seed passages (titles + snippets):\n{seeds}\n\n"
    "Write 3 to 5 short search queries that would retrieve the NEXT missing entities "
    "needed to answer the question (people, places, organizations, radio call signs, "
    "castles, counties, etc.). Prefer concrete noun phrases over repeating the full "
    "question. Include at least one explicit relation query when a person/org is already "
    "known from seeds (e.g. '<name> place of birth', '<name> headquarters', "
    "'<brand> manufacturer'). Do not invent facts not supported by the seeds; you MAY "
    "ask for the entity type implied by the question once seeds have identified the "
    "prior hop.\n"
    'Return {{"queries": ["...", "..."]}} only.'
)


def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"no JSON object in model text: {text[:240]!r}")
    return match.group(0)


async def llm_json(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    retries: int = 4,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 1024,
        }
        if "deepseek" in model.lower():
            payload["reasoning_effort"] = "none"
            payload["thinking"] = False
        if attempt > 1:
            payload["messages"][0]["content"] += (
                " Previous reply was invalid; return only the JSON object."
            )
        try:
            async with session.post(
                endpoint, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=180)
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            if not str(content).strip():
                raise RuntimeError("empty model content")
            return json.loads(_extract_json_object(str(content)))
        except Exception as error:  # noqa: BLE001
            last_error = error
            await asyncio.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"llm_json failed: {last_error}")


def fill_placeholders(template: str, answers: dict[int, str]) -> str:
    out = template
    for idx, ans in answers.items():
        if not ans:
            continue
        out = out.replace(f"{{{{{idx}}}}}", ans)
        out = out.replace(f"{{{idx}}}", ans)
    return out


def rrf_merge(hit_lists: list[list[dict]], *, rrf_k: int = 60, limit: int = 100) -> list[dict]:
    scores: dict[int, float] = defaultdict(float)
    best: dict[int, dict] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            cid = int(hit["id"])
            scores[cid] += 1.0 / (rrf_k + rank)
            best.setdefault(cid, hit)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for cid, score in ranked[:limit]:
        row = dict(best[cid])
        row["rrf_score"] = score
        out.append(row)
    return out


def c6_pool_docids(row: dict) -> set[int]:
    ids: set[int] = set()
    for c in row.get("initial_candidates") or []:
        if c.get("document_id") is not None:
            ids.add(int(c["document_id"]))
    for step in row.get("trace") or []:
        for did in step.get("candidate_document_ids") or []:
            ids.add(int(did))
        # some traces only keep chunk ids; map via initial_candidates when possible
        for cid in step.get("candidate_ids") or []:
            ids.add(int(cid))
    return ids


def c6_seed_snippets(row: dict, chunks: dict[int, dict], *, limit: int = 8) -> list[dict]:
    """Prefer hop0 frontier / high-grade passages as rewrite seeds."""
    id2title = {
        int(c["id"]): c.get("title") or ""
        for c in (row.get("initial_candidates") or [])
    }
    hop0 = next(
        (t for t in row.get("trace") or [] if t.get("stage") == "jev" and t.get("hop") == 0),
        None,
    )
    ordered: list[int] = []
    if hop0:
        ordered.extend(int(x) for x in (hop0.get("frontier_ids") or []))
        for s in hop0.get("scores") or []:
            if s.get("grade", -1) >= 4:
                ordered.append(int(s["id"]))
    # fallback: first initial candidates
    if not ordered:
        ordered = [int(c["id"]) for c in (row.get("initial_candidates") or [])[:limit]]
    seen: set[int] = set()
    seeds: list[dict] = []
    for cid in ordered:
        if cid in seen:
            continue
        seen.add(cid)
        ch = chunks.get(cid)
        title = (ch or {}).get("title") or id2title.get(cid) or str(cid)
        content = ((ch or {}).get("content") or "")[:400]
        seeds.append({"id": cid, "title": title, "content": content})
        if len(seeds) >= limit:
            break
    return seeds


def passages_blob(hits: list[dict], *, n: int = 8) -> str:
    parts = []
    for h in hits[:n]:
        title = h.get("title") or ""
        content = (h.get("content") or "")[:500]
        parts.append(f"- {title}: {content}")
    return "\n".join(parts) if parts else "(none)"


async def retrieve_many(
    retriever: VectorRetriever,
    index: CorpusIndex,
    queries: list[str],
    *,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    hit_lists = []
    traces = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            hit_lists.append([])
            traces.append({"query": q, "hit_n": 0, "top_titles": []})
            continue
        hits, _usage = await retriever.search(index, q, limit=top_k, excluded=set())
        hit_lists.append(hits)
        traces.append(
            {
                "query": q,
                "hit_n": len(hits),
                "top_titles": [h.get("title") for h in hits[:5]],
                "top_doc_ids": [int(h["document_id"]) for h in hits[:10]],
            }
        )
    merged = rrf_merge(hit_lists, limit=100)
    return merged, traces


async def run_iterative(
    session: aiohttp.ClientSession,
    *,
    retriever: VectorRetriever,
    index: CorpusIndex,
    endpoint: str,
    model: str,
    api_key: str,
    sub_questions: list[str],
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    answers: dict[int, str] = {}
    hit_lists: list[list[dict]] = []
    traces: list[dict] = []
    for i, template in enumerate(sub_questions, start=1):
        query = fill_placeholders(template, answers)
        hits, _ = await retriever.search(index, query, limit=top_k, excluded=set())
        hit_lists.append(hits)
        extract = await llm_json(
            session,
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            system=EXTRACT_SYSTEM,
            user=EXTRACT_USER.format(
                sub_question=query, passages=passages_blob(hits)
            ),
        )
        answer = str(extract.get("answer") or "").strip()
        answers[i] = answer
        traces.append(
            {
                "step": i,
                "template": template,
                "query": query,
                "answer": answer,
                "hit_n": len(hits),
                "top_titles": [h.get("title") for h in hits[:5]],
                "top_doc_ids": [int(h["document_id"]) for h in hits[:10]],
            }
        )
    return rrf_merge(hit_lists, limit=100), traces


async def run(args: argparse.Namespace) -> None:
    never_rows = json.loads(args.never.read_text(encoding="utf-8"))
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in never_rows:
        by_qid[row["qid"]].append(row)

    questions: dict[str, str] = {}
    baseline_pools: dict[str, set[int]] = {}
    detail_rows: dict[str, dict] = {}
    for line in args.run.joinpath("details.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["dataset_sample_id"]
        if qid not in by_qid:
            continue
        questions[qid] = row["question"]
        baseline_pools[qid] = c6_pool_docids(row)
        detail_rows[qid] = row

    missing = sorted(set(by_qid) - set(questions))
    if missing:
        raise SystemExit(f"questions missing from run details: {missing}")

    dataset_dir = args.dataset_dir
    corpus, _records, _unit = load_dataset("musique", dataset_dir)
    corpus_sha = corpus_fingerprint(dataset_dir / "musique_corpus.json")
    index = CorpusIndex(
        args.cache_dir / "musique.sqlite",
        corpus,
        corpus_sha,
        DEFAULT_CHUNK_CHARS,
    )
    vector_cache = VectorCache.load(args.cache_dir, "musique")
    vector_cache.validate_index(index, corpus_sha)
    chunks = {
        int(r["id"]): {"id": int(r["id"]), "title": r["title"], "content": r["content"]}
        for r in index.db.execute("SELECT id, title, content FROM chunks")
    }

    endpoint = chat_completions_endpoint(os.environ["SAG_EVIDENCE_BASE_URL"])
    model = args.model or os.environ.get("SAG_EVIDENCE_MODEL", "qwen3.8-27b")
    api_key = os.environ["SAG_EVIDENCE_API_KEY"]
    emb_base = os.environ["SAG_EMBEDDING_BASE_URL"]
    emb_model = os.environ["SAG_EMBEDDING_MODEL"]
    emb_key = os.environ.get("SAG_EMBEDDING_API_KEY", "")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    gold_hits = {
        "baseline": 0,
        "decomp_only": 0,
        "iterative": 0,
        "seed_followup": 0,
        "union_iter_followup": 0,
    }
    total_golds = len(never_rows)

    async with aiohttp.ClientSession() as session:
        emb_client = EmbeddingClient(
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
        retriever = VectorRetriever(emb_client, vector_cache)

        for qi, (qid, golds) in enumerate(sorted(by_qid.items()), start=1):
            question = questions[qid]
            print(f"[{qi}/{len(by_qid)}] {qid}")
            decomp = await llm_json(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                system=DECOMPOSE_SYSTEM,
                user=DECOMPOSE_USER.format(question=question),
            )
            sub_qs = [str(s).strip() for s in (decomp.get("sub_questions") or []) if str(s).strip()]
            if not sub_qs:
                raise RuntimeError(f"empty decomposition for {qid}: {decomp}")

            # also retrieve original question once for decomp_only fusion
            decomp_queries = [question, *sub_qs]
            decomp_merged, decomp_trace = await retrieve_many(
                retriever, index, decomp_queries, top_k=args.top_k
            )
            iter_merged, iter_trace = await run_iterative(
                session,
                retriever=retriever,
                index=index,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                sub_questions=sub_qs,
                top_k=args.top_k,
            )

            seeds = c6_seed_snippets(detail_rows[qid], chunks, limit=8)
            seed_blob = "\n".join(
                f"- {s['title']}: {s['content']}" for s in seeds
            ) or "(none)"
            follow = await llm_json(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                system=FOLLOWUP_SYSTEM,
                user=FOLLOWUP_USER.format(question=question, seeds=seed_blob),
            )
            follow_qs = [
                str(q).strip() for q in (follow.get("queries") or []) if str(q).strip()
            ]
            follow_merged, follow_trace = await retrieve_many(
                retriever, index, follow_qs, top_k=args.top_k
            )
            union_merged = rrf_merge(
                [
                    iter_merged,
                    follow_merged,
                    decomp_merged,
                ],
                limit=150,
            )

            base_ids = baseline_pools[qid]
            decomp_ids = {int(h["document_id"]) for h in decomp_merged}
            iter_ids = {int(h["document_id"]) for h in iter_merged}
            follow_ids = {int(h["document_id"]) for h in follow_merged}
            union_ids = {int(h["document_id"]) for h in union_merged}

            gold_report = []
            for g in golds:
                did = int(g["document_id"])
                entry = {
                    "document_id": did,
                    "title": g["title"],
                    "in_baseline": did in base_ids,
                    "in_decomp_only": did in decomp_ids,
                    "in_iterative": did in iter_ids,
                    "in_seed_followup": did in follow_ids,
                    "in_union_iter_followup": did in union_ids,
                }
                gold_report.append(entry)
                if entry["in_baseline"]:
                    gold_hits["baseline"] += 1
                if entry["in_decomp_only"]:
                    gold_hits["decomp_only"] += 1
                if entry["in_iterative"]:
                    gold_hits["iterative"] += 1
                if entry["in_seed_followup"]:
                    gold_hits["seed_followup"] += 1
                if entry["in_union_iter_followup"]:
                    gold_hits["union_iter_followup"] += 1

            results.append(
                {
                    "qid": qid,
                    "question": question,
                    "sub_questions": sub_qs,
                    "followup_queries": follow_qs,
                    "seed_titles": [s["title"] for s in seeds],
                    "golds": gold_report,
                    "baseline_pool_n": len(base_ids),
                    "decomp_only": {
                        "pool_n": len(decomp_ids),
                        "trace": decomp_trace,
                        "top_titles": [h.get("title") for h in decomp_merged[:10]],
                    },
                    "iterative": {
                        "pool_n": len(iter_ids),
                        "trace": iter_trace,
                        "top_titles": [h.get("title") for h in iter_merged[:10]],
                    },
                    "seed_followup": {
                        "pool_n": len(follow_ids),
                        "trace": follow_trace,
                        "top_titles": [h.get("title") for h in follow_merged[:10]],
                    },
                    "union_iter_followup": {
                        "pool_n": len(union_ids),
                        "top_titles": [h.get("title") for h in union_merged[:10]],
                    },
                }
            )
            hit_i = sum(1 for g in gold_report if g["in_iterative"])
            hit_d = sum(1 for g in gold_report if g["in_decomp_only"])
            hit_f = sum(1 for g in gold_report if g["in_seed_followup"])
            hit_u = sum(1 for g in gold_report if g["in_union_iter_followup"])
            print(
                f"  sub_qs={len(sub_qs)} follow_qs={len(follow_qs)} never_golds={len(golds)}  "
                f"decomp={hit_d} iter={hit_i} follow={hit_f} union={hit_u}"
            )

    summary = {
        "total_never_golds": total_golds,
        "n_questions": len(by_qid),
        "hits": gold_hits,
        "hit_rates": {k: v / total_golds for k, v in gold_hits.items()},
        "gate_iterative_ge_8": gold_hits["iterative"] >= 8,
        "gate_union_ge_8": gold_hits["union_iter_followup"] >= 8,
        "model": model,
        "top_k": args.top_k,
    }
    (args.out / "results.json").write_text(
        json.dumps({"summary": summary, "questions": results}, ensure_ascii=False, indent=2)
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out / 'results.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--never", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, default=Path("../../dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/sag2"))
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
