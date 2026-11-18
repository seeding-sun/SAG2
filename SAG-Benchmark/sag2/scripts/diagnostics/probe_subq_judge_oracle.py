#!/usr/bin/env python3
"""Oracle ceiling: grade never-pool golds against MuSiQue gold decompositions.

If gold hop questions lift grades, the judge is usable and rewrite/extract is the
bottleneck. If even oracle anchors stay at 0, the judge/criterion is still broken.

Also grades against a filled hop question built from gold answers:
  replace #1/#2 with prior gold answers when present.
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
    state = {"original_question": question, "previously_retrieved_evidence": []}
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
        raise KeyError(document_id)
    title = rows[0]["title"] or ""
    best = max(rows, key=lambda r: len(r["content"] or ""))
    return title, best["content"] or ""


def fill_gold_decomp(decomp: list[dict]) -> list[str]:
    """Build readable hop questions using gold answers for #1/#2/... refs."""
    answers: dict[int, str] = {}
    filled: list[str] = []
    for i, step in enumerate(decomp, start=1):
        q = str(step.get("question") or "")
        # MuSiQue uses #1, #2 in later hops
        out = q
        for idx, ans in answers.items():
            out = out.replace(f"#{idx}", ans)
        # also handle ">> relation" shorthand: "X >> place of birth"
        if ">>" in out and not out.strip().lower().startswith(("who", "what", "when", "where", "which", "how")):
            left, _, right = out.partition(">>")
            out = f"What is the {right.strip()} of {left.strip()}?"
        filled.append(out.strip())
        answers[i] = str(step.get("answer") or "").strip()
    return filled


def title_to_doc_id(index: CorpusIndex, title: str) -> int | None:
    row = index.db.execute(
        "SELECT document_id FROM chunks WHERE title = ? LIMIT 1", (title,)
    ).fetchone()
    return int(row["document_id"]) if row else None


async def run(args: argparse.Namespace) -> None:
    probe = json.loads(args.probe_results.read_text(encoding="utf-8"))
    gold_pool = {
        (q["qid"], int(g["document_id"])): g
        for q in probe["questions"]
        for g in q["golds"]
        if g.get("in_union_iter_followup")
    }

    musique = {
        row["id"]: row
        for row in json.loads(args.musique_json.read_text(encoding="utf-8"))
    }

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

    # Map supporting titles -> doc ids for matching which hop owns which gold.
    work = []
    for (qid, did), g in sorted(gold_pool.items()):
        sample = musique.get(qid)
        if not sample:
            continue
        decomp = sample.get("question_decomposition") or []
        filled = fill_gold_decomp(decomp)
        # hop questions whose supporting paragraph title matches this gold title
        own_hops = []
        title = g.get("title") or ""
        for i, step in enumerate(decomp):
            idx = step.get("paragraph_support_idx")
            paras = sample.get("paragraphs") or []
            if idx is None or not (0 <= int(idx) < len(paras)):
                continue
            if (paras[int(idx)].get("title") or "") == title:
                own_hops.append(filled[i])
        # fallback: all filled hops
        anchors_own = own_hops or filled
        title_text, passage = load_doc_text(index, did)
        work.append(
            {
                "qid": qid,
                "question": sample["question"],
                "answer": sample.get("answer"),
                "document_id": did,
                "title": title_text or title,
                "passage": passage,
                "gold_filled_all": filled,
                "gold_filled_own": anchors_own,
                "our_iter_wrong_example": None,
            }
        )

    print(f"cases={len(work)} model={model}", flush=True)
    results = []
    async with aiohttp.ClientSession() as session:
        for i, case in enumerate(work, start=1):
            print(f"[{i}/{len(work)}] {case['qid']} :: {case['title'][:55]}", flush=True)
            orig = await grade_one(
                session,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                question=case["question"],
                cand_id=case["document_id"],
                title=case["title"],
                passage=case["passage"],
            )
            own_grades = []
            for aq in case["gold_filled_own"]:
                g = await grade_one(
                    session,
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    question=aq,
                    cand_id=case["document_id"],
                    title=case["title"],
                    passage=case["passage"],
                )
                own_grades.append({"anchor": aq, "grade": g})
            all_grades = []
            for aq in case["gold_filled_all"]:
                g = await grade_one(
                    session,
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    question=aq,
                    cand_id=case["document_id"],
                    title=case["title"],
                    passage=case["passage"],
                )
                all_grades.append({"anchor": aq, "grade": g})
            own_max = max((x["grade"] for x in own_grades), default=0)
            all_max = max((x["grade"] for x in all_grades), default=0)
            row = {
                "qid": case["qid"],
                "title": case["title"],
                "document_id": case["document_id"],
                "question": case["question"],
                "grade_orig": orig,
                "grade_oracle_own_max": own_max,
                "grade_oracle_all_max": all_max,
                "own_grades": own_grades,
                "all_grades": all_grades,
            }
            results.append(row)
            print(
                f"  orig={orig} oracle_own={own_max} oracle_all={all_max} "
                f"own={case['gold_filled_own']}",
                flush=True,
            )

    n = len(results) or 1
    summary = {
        "model": model,
        "n": len(results),
        "frac_ge4_orig": sum(r["grade_orig"] >= 4 for r in results) / n,
        "frac_ge4_oracle_own": sum(r["grade_oracle_own_max"] >= 4 for r in results) / n,
        "frac_ge4_oracle_all": sum(r["grade_oracle_all_max"] >= 4 for r in results) / n,
        "n_orig0_oracle_own_ge4": sum(
            1 for r in results if r["grade_orig"] == 0 and r["grade_oracle_own_max"] >= 4
        ),
        "read": (
            "High oracle_own lift ⇒ invest in rewrite/extract fidelity to gold hops. "
            "Low oracle_own ⇒ judge/criterion still fails even with perfect sub-questions."
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
    p.add_argument("--musique-json", type=Path, default=Path("../../dataset/musique.json"))
    p.add_argument("--out", type=Path, default=Path("../../output/probe-subq-judge-oracle"))
    p.add_argument("--dataset-dir", type=Path, default=Path("../../dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/sag2"))
    p.add_argument("--model", default=None)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
