#!/usr/bin/env python3
"""Probe: grade never-pool golds against original Q vs hop sub-questions.

Hypothesis: multi-hop bridge passages look irrelevant to the full original
question (grade 0) but should score higher against the sub-question they serve.
Also check whether non-gold distractors inflate under the same rule.

Arms per candidate:
  - orig:          grade vs original question (empty evidence)
  - sub_max:       max grade over filled iterative sub-queries
  - blend07:       round(0.7 * sub_max + 0.3 * orig)   # user's "原问只占一定比例"
  - blend05:       round(0.5 * sub_max + 0.5 * orig)

Usage (from sag2/):
  set -a; . ./.env; set +a
  .venv/bin/python scripts/diagnostics/probe_subq_judge.py \\
    --probe-results ../../output/probe-never-multiquery-v2-27b/results.json \\
    --mq-run ../../output/probe-never-multiquery-v2-27b/C6-MQ-H1-never11 \\
    --out ../../output/probe-subq-judge
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
    retries: int = 4,
) -> int:
    state = {
        "original_question": question,
        "previously_retrieved_evidence": [],
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


def blend(sub_max: int, orig: int, alpha: float) -> int:
    """alpha = weight on sub_max; round to nearest even-grade bucket used by rubric."""
    raw = alpha * sub_max + (1.0 - alpha) * orig
    # snap to nearest allowed grade in GRADE_RUBRIC
    allowed = sorted(GRADE_RUBRIC)
    return min(allowed, key=lambda g: (abs(g - raw), -g))


def pick_distractors(row: dict, gold_dids: set[int], *, n: int = 2) -> list[dict]:
    """High-ranked non-gold docs from MQ run final candidates / initial pool."""
    out: list[dict] = []
    seen: set[int] = set()
    for c in (row.get("candidates") or []) + (row.get("initial_candidates") or []):
        did = c.get("document_id")
        if did is None:
            continue
        did = int(did)
        if did in gold_dids or did in seen:
            continue
        seen.add(did)
        out.append(
            {
                "document_id": did,
                "chunk_id": int(c["id"]),
                "title": c.get("title") or "",
                "passage": c.get("content") or "",
                "kind": "distractor",
            }
        )
        if len(out) >= n:
            break
    return out


def load_doc_text(index: CorpusIndex, document_id: int) -> tuple[str, str]:
    """Return (title, concatenated chunk text) for a corpus document_id."""
    rows = list(
        index.db.execute(
            "SELECT id, title, content FROM chunks WHERE document_id = ? ORDER BY id",
            (document_id,),
        )
    )
    if not rows:
        raise KeyError(f"document_id {document_id} missing from chunk index")
    title = rows[0]["title"] or ""
    # Prefer the longest chunk as the judged passage (typical wiki-style one chunk/doc).
    best = max(rows, key=lambda r: len(r["content"] or ""))
    return title, best["content"] or ""


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

    # Focus: golds that MQ retrieval put in pool (union hit) — the "捞进来了" slice.
    work: list[dict] = []
    for qid, qrow in sorted(questions.items()):
        mq = mq_rows.get(qid)
        if mq is None:
            continue
        filled_subs: list[str] = []
        answers: dict[int, str] = {}
        for step in (qrow.get("iterative") or {}).get("trace") or []:
            answers[int(step["step"])] = str(step.get("answer") or "").strip()
        for i, template in enumerate(qrow.get("sub_questions") or [], start=1):
            filled = fill_placeholders(template, answers)
            # Drop still-open placeholders — those are decomp_only junk queries.
            if re.search(r"\{\{?\d+\}?\}", filled):
                continue
            filled_subs.append(filled)
        if not filled_subs:
            # fallback: raw sub_questions without braces only
            filled_subs = [
                s
                for s in (qrow.get("sub_questions") or [])
                if not re.search(r"\{\{?\d+\}?\}", s)
            ]

        gold_dids = {
            int(g["document_id"])
            for g in qrow["golds"]
            if g.get("in_union_iter_followup")
        }
        distractors = pick_distractors(mq, gold_dids, n=args.distractors_per_q)

        for g in qrow["golds"]:
            if not g.get("in_union_iter_followup"):
                continue
            did = int(g["document_id"])
            title, passage = load_doc_text(index, did)
            work.append(
                {
                    "qid": qid,
                    "question": qrow["question"],
                    "sub_questions_filled": filled_subs,
                    "document_id": did,
                    "title": title or g.get("title") or "",
                    "passage": passage,
                    "kind": "gold",
                    "cand_id": did,
                }
            )
        for d in distractors:
            if not d["passage"]:
                title, passage = load_doc_text(index, d["document_id"])
                d["title"] = title or d["title"]
                d["passage"] = passage
            work.append(
                {
                    "qid": qid,
                    "question": qrow["question"],
                    "sub_questions_filled": filled_subs,
                    "document_id": d["document_id"],
                    "title": d["title"],
                    "passage": d["passage"],
                    "kind": "distractor",
                    "cand_id": d["document_id"],
                }
            )

    print(f"cases={len(work)} model={model}", flush=True)
    results = []
    async with aiohttp.ClientSession() as session:
        for i, case in enumerate(work, start=1):
            print(
                f"[{i}/{len(work)}] {case['kind']} {case['qid']} :: {case['title'][:60]}",
                flush=True,
            )
            orig = await grade_one(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                question=case["question"],
                cand_id=case["cand_id"],
                title=case["title"],
                passage=case["passage"],
            )
            sub_grades = []
            for sq in case["sub_questions_filled"]:
                g = await grade_one(
                    session,
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    question=sq,
                    cand_id=case["cand_id"],
                    title=case["title"],
                    passage=case["passage"],
                )
                sub_grades.append({"sub_question": sq, "grade": g})
            sub_max = max((s["grade"] for s in sub_grades), default=0)
            row = {
                **{k: case[k] for k in ("qid", "question", "document_id", "title", "kind")},
                "sub_questions_filled": case["sub_questions_filled"],
                "grade_orig": orig,
                "sub_grades": sub_grades,
                "grade_sub_max": sub_max,
                "grade_blend07": blend(sub_max, orig, 0.7),
                "grade_blend05": blend(sub_max, orig, 0.5),
                "lift_sub_max": sub_max - orig,
                "lift_blend07": blend(sub_max, orig, 0.7) - orig,
            }
            results.append(row)
            print(
                f"  orig={orig} sub_max={sub_max} blend07={row['grade_blend07']} "
                f"subs={[s['grade'] for s in sub_grades]}",
                flush=True,
            )

    def summarize(kind: str) -> dict:
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            return {"n": 0}
        def rate(key: str, thr: int = 4) -> float:
            return sum(1 for r in rows if r[key] >= thr) / len(rows)

        return {
            "n": len(rows),
            "mean_orig": sum(r["grade_orig"] for r in rows) / len(rows),
            "mean_sub_max": sum(r["grade_sub_max"] for r in rows) / len(rows),
            "mean_blend07": sum(r["grade_blend07"] for r in rows) / len(rows),
            "frac_ge4_orig": rate("grade_orig"),
            "frac_ge4_sub_max": rate("grade_sub_max"),
            "frac_ge4_blend07": rate("grade_blend07"),
            "n_lift_sub_max": sum(1 for r in rows if r["lift_sub_max"] > 0),
            "n_drop_sub_max": sum(1 for r in rows if r["lift_sub_max"] < 0),
            "n_orig0_sub_ge4": sum(
                1 for r in rows if r["grade_orig"] == 0 and r["grade_sub_max"] >= 4
            ),
        }

    summary = {
        "model": model,
        "gold": summarize("gold"),
        "distractor": summarize("distractor"),
        "read": (
            "If gold frac_ge4 rises under sub_max/blend while distractor stays flat, "
            "sub-question anchoring helps the in-pool bottleneck; if distractors also "
            "inflate, the rule is too loose."
        ),
    }
    (args.out / "results.json").write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


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
    p.add_argument("--out", type=Path, default=Path("../../output/probe-subq-judge"))
    p.add_argument("--dataset-dir", type=Path, default=Path("../../dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/sag2"))
    p.add_argument("--model", default=None)
    p.add_argument("--distractors-per-q", type=int, default=2)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
