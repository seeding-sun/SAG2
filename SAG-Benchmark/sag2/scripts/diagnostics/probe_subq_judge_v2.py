#!/usr/bin/env python3
"""Sub-question judge v2: grade against the query that actually retrieved the doc.

v1 only scored filled iterative sub-questions. Failure autopsy showed many never
golds entered the pool via seed_followup queries, so the grade anchor was wrong.

Arms:
  - orig:                 original multi-hop question
  - iter_max:             max over filled iterative sub-questions (v1)
  - follow_max:           max over seed_followup rewrite queries
  - retrieve_max:         max over queries whose top hits contained this doc
  - all_max:              max(iter, follow, retrieve)
  - blend07_all:          snap(0.7*all_max + 0.3*orig)
  - *_ev:                 same arms with hop0 frontier seeds as evidence

Usage (from sag2/):
  set -a; . ./.env; set +a
  .venv/bin/python scripts/diagnostics/probe_subq_judge_v2.py \\
    --out ../../output/probe-subq-judge-v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_evidence_scorer import chat_completions_endpoint, parse_grades  # noqa: E402
from run_sag2_benchmark import load_dataset  # noqa: E402
from sag2 import (  # noqa: E402
    DEFAULT_CHUNK_CHARS,
    GRADE_CRITERION,
    GRADE_RUBRIC,
    CorpusIndex,
    compact_json,
    corpus_fingerprint,
)


def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"no JSON object in model text: {text[:240]!r}")
    return match.group(0)


def fill_placeholders(template: str, answers: dict[int, str]) -> str:
    out = template
    for idx, ans in answers.items():
        if not ans:
            continue
        out = out.replace(f"{{{{{idx}}}}}", ans)
        out = out.replace(f"{{{idx}}}", ans)
    return out


def snap_blend(sub_max: int, orig: int, alpha: float) -> int:
    raw = alpha * sub_max + (1.0 - alpha) * orig
    allowed = sorted(GRADE_RUBRIC)
    return min(allowed, key=lambda g: (abs(g - raw), -g))


async def grade_one(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    question: str,
    cand_id: int,
    title: str,
    passage: str,
    evidence: list[dict],
    retries: int = 4,
) -> int:
    state = {
        "original_question": question,
        "previously_retrieved_evidence": [
            {
                "passage_id": e["id"],
                "title": e["title"],
                "passage": e["content"],
            }
            for e in evidence
        ],
    }
    key = f"candidate_{cand_id}"
    cand_obj = {key: {"title": title, "passage": passage[:1200]}}
    system = (
        "You are a strict evidence judge. Return only one JSON object with a "
        f"'grades' object. Each candidate id must appear exactly once and map to an "
        f"integer grade from {sorted(GRADE_RUBRIC)}. Do not add explanations or keys."
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": compact_json(
                        {
                            "criterion": GRADE_CRITERION,
                            "shared_context": state,
                            "candidates": cand_obj,
                        }
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        if "deepseek" in model.lower():
            payload["reasoning_effort"] = "none"
            payload["thinking"] = False
        if attempt > 1:
            payload["messages"][0]["content"] += (
                " The previous response was empty or invalid JSON; return only the grades object."
            )
        try:
            async with session.post(
                endpoint, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            if not str(content).strip():
                raise RuntimeError("empty model content")
            grades = parse_grades(_extract_json_object(str(content)), {key}, fill_missing=0)
            return int(grades[key])
        except Exception as error:  # noqa: BLE001
            last_error = error
            await asyncio.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"grade_one failed: {last_error}")


def load_doc_text(index: CorpusIndex, document_id: int) -> tuple[str, str]:
    rows = list(
        index.db.execute(
            "SELECT id, title, content FROM chunks WHERE document_id = ? ORDER BY id",
            (document_id,),
        )
    )
    if not rows:
        raise KeyError(f"document_id {document_id} missing")
    title = rows[0]["title"] or ""
    best = max(rows, key=lambda r: len(r["content"] or ""))
    return title, best["content"] or ""


def filled_iterative(qrow: dict) -> list[str]:
    answers: dict[int, str] = {}
    for step in (qrow.get("iterative") or {}).get("trace") or []:
        answers[int(step["step"])] = str(step.get("answer") or "").strip()
    out = []
    for template in qrow.get("sub_questions") or []:
        filled = fill_placeholders(template, answers)
        if re.search(r"\{\{?\d+\}?\}", filled):
            continue
        out.append(filled)
    return out


def queries_that_hit(qrow: dict, document_id: int, *, top_n: int = 30) -> list[str]:
    """Return retrieval queries whose top_n doc ids contain document_id."""
    hits: list[str] = []
    seen: set[str] = set()
    for arm in ("decomp_only", "iterative", "seed_followup"):
        for t in (qrow.get(arm) or {}).get("trace") or []:
            tops = [int(x) for x in (t.get("top_doc_ids") or [])[:top_n]]
            q = (t.get("query") or "").strip()
            if not q or document_id not in tops:
                continue
            if q in seen:
                continue
            seen.add(q)
            hits.append(q)
    return hits


def mq_seed_evidence(mq_row: dict, *, limit: int = 5) -> list[dict]:
    """Use hop0 frontier / high-grade passages as evidence pack (exclude later)."""
    hop0 = next(
        (t for t in mq_row.get("trace") or [] if t.get("stage") == "jev" and t.get("hop") == 0),
        None,
    )
    id2cand = {
        int(c["id"]): c
        for c in (mq_row.get("initial_candidates") or []) + (mq_row.get("candidates") or [])
    }
    ordered: list[int] = []
    if hop0:
        ordered.extend(int(x) for x in (hop0.get("frontier_ids") or []))
        for s in hop0.get("scores") or []:
            if int(s.get("grade") or -1) >= 4:
                ordered.append(int(s["id"]))
    if not ordered:
        ordered = [int(c["id"]) for c in (mq_row.get("initial_candidates") or [])[:limit]]
    out = []
    seen: set[int] = set()
    for cid in ordered:
        if cid in seen:
            continue
        seen.add(cid)
        c = id2cand.get(cid)
        if not c:
            continue
        out.append(
            {
                "id": cid,
                "title": c.get("title") or "",
                "content": (c.get("content") or "")[:400],
            }
        )
        if len(out) >= limit:
            break
    return out


def pick_distractors(mq_row: dict, gold_dids: set[int], *, n: int = 2) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for c in (mq_row.get("candidates") or []) + (mq_row.get("initial_candidates") or []):
        did = c.get("document_id")
        if did is None:
            continue
        did = int(did)
        if did in gold_dids or did in seen:
            continue
        seen.add(did)
        out.append(did)
        if len(out) >= n:
            break
    return out


async def grade_anchors(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    cand_id: int,
    title: str,
    passage: str,
    anchors: list[str],
    evidence: list[dict],
    cache: dict[tuple, int],
) -> list[dict]:
    rows = []
    for q in anchors:
        key = (q, cand_id, tuple(e["id"] for e in evidence))
        if key in cache:
            g = cache[key]
        else:
            g = await grade_one(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                question=q,
                cand_id=cand_id,
                title=title,
                passage=passage,
                evidence=evidence,
            )
            cache[key] = g
        rows.append({"anchor": q, "grade": g})
    return rows


async def run(args: argparse.Namespace) -> None:
    probe = json.loads(args.probe_results.read_text(encoding="utf-8"))
    questions = {q["qid"]: q for q in probe["questions"]}
    mq_rows: dict[str, dict] = {}
    for line in args.mq_run.joinpath("details.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mq_rows[row["dataset_sample_id"]] = row

    dataset_dir = args.dataset_dir
    corpus, _records, _unit = load_dataset("musique", dataset_dir)
    corpus_sha = corpus_fingerprint(dataset_dir / "musique_corpus.json")
    index = CorpusIndex(
        args.cache_dir / "musique.sqlite",
        corpus,
        corpus_sha,
        DEFAULT_CHUNK_CHARS,
    )

    endpoint = chat_completions_endpoint(os.environ["SAG_EVIDENCE_BASE_URL"])
    model = args.model or os.environ.get("SAG_EVIDENCE_MODEL", "qwen3.8-27b")
    api_key = os.environ["SAG_EVIDENCE_API_KEY"]
    args.out.mkdir(parents=True, exist_ok=True)

    work: list[dict] = []
    for qid, qrow in sorted(questions.items()):
        mq = mq_rows.get(qid)
        if mq is None:
            continue
        iters = filled_iterative(qrow)
        follows = [str(x).strip() for x in (qrow.get("followup_queries") or []) if str(x).strip()]
        gold_dids = {
            int(g["document_id"])
            for g in qrow["golds"]
            if g.get("in_union_iter_followup")
        }
        seeds = mq_seed_evidence(mq, limit=5)

        def make_case(did: int, kind: str) -> dict:
            title, passage = load_doc_text(index, did)
            retrieve_qs = queries_that_hit(qrow, did)
            return {
                "qid": qid,
                "question": qrow["question"],
                "kind": kind,
                "document_id": did,
                "title": title,
                "passage": passage,
                "iter_anchors": iters,
                "follow_anchors": follows,
                "retrieve_anchors": retrieve_qs,
                "evidence": [
                    e
                    for e in seeds
                    if int(e["id"]) != did  # cheap self-guard by doc chunk id may differ; title filter below
                    and (e.get("title") or "") != title
                ],
            }

        for did in sorted(gold_dids):
            work.append(make_case(did, "gold"))
        for did in pick_distractors(mq, gold_dids, n=args.distractors_per_q):
            work.append(make_case(did, "distractor"))

    print(f"cases={len(work)} model={model}", flush=True)
    cache: dict[tuple, int] = {}
    results = []

    async with aiohttp.ClientSession() as session:
        for i, case in enumerate(work, start=1):
            print(
                f"[{i}/{len(work)}] {case['kind']} {case['qid']} :: {case['title'][:55]}",
                flush=True,
            )
            orig = await grade_one(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                question=case["question"],
                cand_id=case["document_id"],
                title=case["title"],
                passage=case["passage"],
                evidence=[],
            )
            iter_rows = await grade_anchors(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                cand_id=case["document_id"],
                title=case["title"],
                passage=case["passage"],
                anchors=case["iter_anchors"],
                evidence=[],
                cache=cache,
            )
            follow_rows = await grade_anchors(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                cand_id=case["document_id"],
                title=case["title"],
                passage=case["passage"],
                anchors=case["follow_anchors"],
                evidence=[],
                cache=cache,
            )
            retrieve_rows = await grade_anchors(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                cand_id=case["document_id"],
                title=case["title"],
                passage=case["passage"],
                anchors=case["retrieve_anchors"],
                evidence=[],
                cache=cache,
            )
            # evidence variants for the best non-orig anchors only (save calls):
            # regrade retrieve anchors + follow anchors with evidence if empty-ev all_max < 4
            iter_max = max((r["grade"] for r in iter_rows), default=0)
            follow_max = max((r["grade"] for r in follow_rows), default=0)
            retrieve_max = max((r["grade"] for r in retrieve_rows), default=0)
            all_max = max(iter_max, follow_max, retrieve_max)

            ev_retrieve_rows: list[dict] = []
            ev_follow_rows: list[dict] = []
            ev_iter_rows: list[dict] = []
            if case["evidence"] and all_max < 4:
                # only spend evidence calls when empty-ev failed to promote
                union_anchors = []
                seen_a: set[str] = set()
                for a in case["retrieve_anchors"] + case["follow_anchors"] + case["iter_anchors"]:
                    if a not in seen_a:
                        seen_a.add(a)
                        union_anchors.append(a)
                ev_all = await grade_anchors(
                    session,
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    cand_id=case["document_id"],
                    title=case["title"],
                    passage=case["passage"],
                    anchors=union_anchors,
                    evidence=case["evidence"],
                    cache=cache,
                )
                ev_map = {r["anchor"]: r["grade"] for r in ev_all}
                ev_retrieve_rows = [
                    {"anchor": a, "grade": ev_map[a]}
                    for a in case["retrieve_anchors"]
                    if a in ev_map
                ]
                ev_follow_rows = [
                    {"anchor": a, "grade": ev_map[a]} for a in case["follow_anchors"] if a in ev_map
                ]
                ev_iter_rows = [
                    {"anchor": a, "grade": ev_map[a]} for a in case["iter_anchors"] if a in ev_map
                ]
            ev_iter_max = max((r["grade"] for r in ev_iter_rows), default=iter_max)
            ev_follow_max = max((r["grade"] for r in ev_follow_rows), default=follow_max)
            ev_retrieve_max = max((r["grade"] for r in ev_retrieve_rows), default=retrieve_max)
            # If we skipped evidence (already >=4), treat ev_* as equal to empty-ev.
            if all_max >= 4:
                ev_iter_max, ev_follow_max, ev_retrieve_max = iter_max, follow_max, retrieve_max
            ev_all_max = max(ev_iter_max, ev_follow_max, ev_retrieve_max, all_max)

            row = {
                "qid": case["qid"],
                "question": case["question"],
                "kind": case["kind"],
                "document_id": case["document_id"],
                "title": case["title"],
                "retrieve_anchors": case["retrieve_anchors"],
                "grade_orig": orig,
                "grade_iter_max": iter_max,
                "grade_follow_max": follow_max,
                "grade_retrieve_max": retrieve_max,
                "grade_all_max": all_max,
                "grade_blend07_all": snap_blend(all_max, orig, 0.7),
                "grade_ev_all_max": ev_all_max,
                "grade_blend07_ev": snap_blend(ev_all_max, orig, 0.7),
                "iter_grades": iter_rows,
                "follow_grades": follow_rows,
                "retrieve_grades": retrieve_rows,
                "evidence_n": len(case["evidence"]),
                "evidence_used": bool(case["evidence"] and all_max < 4),
            }
            results.append(row)
            print(
                f"  orig={orig} iter={iter_max} follow={follow_max} "
                f"retrieve={retrieve_max} all={all_max} ev_all={ev_all_max} "
                f"retr_q={case['retrieve_anchors'][:2]}",
                flush=True,
            )

    def summarize(kind: str) -> dict:
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            return {"n": 0}

        def frac(key: str, thr: int = 4) -> float:
            return sum(1 for r in rows if r[key] >= thr) / len(rows)

        return {
            "n": len(rows),
            "frac_ge4_orig": frac("grade_orig"),
            "frac_ge4_iter_max": frac("grade_iter_max"),
            "frac_ge4_follow_max": frac("grade_follow_max"),
            "frac_ge4_retrieve_max": frac("grade_retrieve_max"),
            "frac_ge4_all_max": frac("grade_all_max"),
            "frac_ge4_blend07_all": frac("grade_blend07_all"),
            "frac_ge4_ev_all_max": frac("grade_ev_all_max"),
            "frac_ge4_blend07_ev": frac("grade_blend07_ev"),
            "n_orig0_all_ge4": sum(
                1 for r in rows if r["grade_orig"] == 0 and r["grade_all_max"] >= 4
            ),
            "n_orig0_ev_ge4": sum(
                1 for r in rows if r["grade_orig"] == 0 and r["grade_ev_all_max"] >= 4
            ),
            "mean_orig": sum(r["grade_orig"] for r in rows) / len(rows),
            "mean_all_max": sum(r["grade_all_max"] for r in rows) / len(rows),
            "mean_ev_all_max": sum(r["grade_ev_all_max"] for r in rows) / len(rows),
        }

    summary = {
        "model": model,
        "gold": summarize("gold"),
        "distractor": summarize("distractor"),
        "design_note": (
            "Production implication: when multi-query retrieves doc D via query q_i, "
            "judge D against q_i (or max(q_i, orig)), not only against the full original question."
        ),
    }
    (args.out / "results.json").write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gate = {
        "gold_ge4_all_max_ge_5_of_10": summary["gold"].get("frac_ge4_all_max", 0) >= 0.5,
        "distractor_ge4_all_max_not_worse_than_orig_plus_15pp": (
            summary["distractor"].get("frac_ge4_all_max", 1)
            <= summary["distractor"].get("frac_ge4_orig", 0) + 0.15
        ),
    }
    (args.out / "GATE.md").write_text(
        "# Subq judge v2 gate\n\n"
        + json.dumps({"summary": summary, "gate": gate}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("GATE", gate, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--probe-results",
        type=Path,
        default=Path("../../output/probe-never-multiquery-v2-27b/results.json"),
    )
    p.add_argument(
        "--mq-run",
        type=Path,
        default=Path("../../output/probe-never-multiquery-v2-27b/C6-MQ-H1-never11"),
    )
    p.add_argument("--out", type=Path, default=Path("../../output/probe-subq-judge-v2"))
    p.add_argument("--dataset-dir", type=Path, default=Path("../../dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/sag2"))
    p.add_argument("--model", default=None)
    p.add_argument("--distractors-per-q", type=int, default=2)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
