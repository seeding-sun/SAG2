#!/usr/bin/env python3
"""Frozen-pool three-arm probe: orig / self-fill / oracle on the same candidate ids.

Freeze hop0 multi-query pools from an existing C6-MQ-H1 run, then re-score only
(no re-retrieve, no expansion). Self-fill anchors come from a dual-arm run's
templates+answers; oracle anchors from MuSiQue gold question_decomposition.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from hop_anchor import fill_placeholders, has_open_placeholder  # noqa: E402
from qwen_evidence_scorer import QwenEvidenceScorer  # noqa: E402
from run_hybrid_compare import load_subset  # noqa: E402
from run_sag2_benchmark import load_dataset, write_json  # noqa: E402
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, corpus_fingerprint  # noqa: E402


def fill_gold_decomp(decomp: list[dict]) -> list[str]:
    answers: dict[int, str] = {}
    filled: list[str] = []
    for i, step in enumerate(decomp, start=1):
        q = str(step.get("question") or "")
        out = q
        for idx, ans in answers.items():
            out = out.replace(f"#{idx}", ans)
        if ">>" in out and not out.strip().lower().startswith(
            ("who", "what", "when", "where", "which", "how")
        ):
            left, _, right = out.partition(">>")
            out = f"What is the {right.strip()} of {left.strip()}?"
        filled.append(out.strip())
        answers[i] = str(step.get("answer") or "").strip()
    return [f for f in filled if f]


def self_fill_anchors(templates: list[str] | None, answers: dict | None) -> list[str]:
    if not templates:
        return []
    ans = {int(k): str(v) for k, v in (answers or {}).items()}
    out: list[str] = []
    for tmpl in templates:
        filled = fill_placeholders(tmpl, ans)
        if has_open_placeholder(filled):
            continue
        if filled and filled not in out:
            out.append(filled)
    return out


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["dataset_sample_id"]] = row
    return rows


def chunks_for_ids(index: CorpusIndex, ids: list[int]) -> list[dict]:
    out = []
    for cid in ids:
        row = index.db.execute(
            "SELECT id, document_id, title, content, full_doc, source_id FROM chunks WHERE id = ?",
            (int(cid),),
        ).fetchone()
        if not row:
            continue
        out.append(
            {
                "id": int(row["id"]),
                "document_id": int(row["document_id"]),
                "title": row["title"] or "",
                "content": row["content"] or "",
                "full_doc": row["full_doc"] or "",
                "source_id": row["source_id"],
            }
        )
    return out


def recall_at_k(ranked_full_docs: list[str], gold: list[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    top = set(ranked_full_docs[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def pool_hit(pool_full_docs: set[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for g in gold if g in pool_full_docs) / len(gold)


async def score_max_anchors(
    scorer: QwenEvidenceScorer,
    *,
    anchors: list[str],
    candidates: list[dict],
) -> list[dict]:
    """Score against each anchor; keep per-id max grade."""
    if not candidates:
        return []
    best: dict[int, dict] = {}
    for anchor in anchors:
        rows, _ = await scorer.score(anchor, candidates, evidence=[])
        for row in rows:
            cid = int(row["id"])
            g = int(row.get("jev_grade", round(float(row["jev_score"]) * 10)))
            prev = best.get(cid)
            if prev is None or g > int(prev.get("jev_grade", 0)):
                new = dict(row)
                new["jev_grade"] = g
                new["jev_score"] = g / 10.0
                best[cid] = new
    # preserve candidate metadata
    id2c = {int(c["id"]): c for c in candidates}
    merged = []
    for cid, row in best.items():
        base = dict(id2c[cid])
        base.update(row)
        merged.append(base)
    merged.sort(key=lambda c: (-c["jev_score"], -c.get("fusion_score", 0), c["id"]))
    return merged


async def run(args: argparse.Namespace) -> None:
    mq_rows = load_jsonl(args.mq_run / "details.jsonl")
    dual_rows = load_jsonl(args.dual_run / "details.jsonl")
    musique = {
        row["id"]: row
        for row in json.loads(args.musique_json.read_text(encoding="utf-8"))
    }

    corpus, records, unit = load_dataset("musique", args.dataset_dir)
    selected = load_subset(args.sample_file, records)
    by_id = {r["id"]: r for r in selected}

    corpus_sha = corpus_fingerprint(args.dataset_dir / "musique_corpus.json")
    index = CorpusIndex(
        args.cache_dir / "musique.sqlite", corpus, corpus_sha, DEFAULT_CHUNK_CHARS
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model = args.evidence_model
    api_key = os.environ["SAG_EVIDENCE_API_KEY"]

    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        scorer = QwenEvidenceScorer(
            session,
            os.environ["SAG_EVIDENCE_BASE_URL"],
            model,
            api_key,
            journal_dir=out / "requests",
            batch_size=args.evidence_batch_size,
            concurrency=1,
            response_format=True,
            scoring_scale="grade",
            fill_missing_score=0,
            max_output_tokens_per_candidate=64,
            strict_model=True,
        )
        ordered_ids = [r["id"] for r in selected]
        for i, qid in enumerate(ordered_ids, start=1):
            mq = mq_rows[qid]
            dual = dual_rows.get(qid) or {}
            record = by_id[qid]
            gold = record["gold_docs"]
            question = record["question"]

            pool_ids = list(mq.get("initial_candidate_ids") or [])
            # Also include any hop1-only ids that made final retrieved set, so golds
            # that only arrived via expansion stay visible when present in MQ arm.
            for t in mq.get("trace") or []:
                if t.get("stage") == "jev":
                    for s in t.get("scores") or []:
                        cid = int(s["id"])
                        if cid not in pool_ids:
                            pool_ids.append(cid)
            candidates = chunks_for_ids(index, pool_ids)
            for rank, c in enumerate(candidates):
                c["fusion_score"] = 1.0 / (61 + rank)

            self_anchors = self_fill_anchors(
                dual.get("hop_anchor_templates"),
                {
                    int(k): v
                    for k, v in (dual.get("hop_anchor_answers") or {}).items()
                },
            )
            sample = musique.get(qid) or {}
            oracle_anchors = fill_gold_decomp(sample.get("question_decomposition") or [])

            print(
                f"[{i}/{len(ordered_ids)}] {qid} pool={len(candidates)} "
                f"self_n={len(self_anchors)} oracle_n={len(oracle_anchors)}"
            )

            orig_ranked = await score_max_anchors(
                scorer, anchors=[question], candidates=candidates
            )
            self_ranked = await score_max_anchors(
                scorer,
                anchors=[question, *self_anchors] if self_anchors else [question],
                candidates=candidates,
            )
            oracle_ranked = await score_max_anchors(
                scorer,
                anchors=[question, *oracle_anchors] if oracle_anchors else [question],
                candidates=candidates,
            )

            pool_docs = {c["full_doc"] for c in candidates}
            row = {
                "dataset_sample_id": qid,
                "question": question,
                "pool_size": len(candidates),
                "pool_gold_recall": pool_hit(pool_docs, gold),
                "self_anchors": self_anchors,
                "oracle_anchors": oracle_anchors,
                "metrics": {
                    "orig": {
                        "Recall@10": recall_at_k(
                            [c["full_doc"] for c in orig_ranked], gold, 10
                        )
                    },
                    "self": {
                        "Recall@10": recall_at_k(
                            [c["full_doc"] for c in self_ranked], gold, 10
                        )
                    },
                    "oracle": {
                        "Recall@10": recall_at_k(
                            [c["full_doc"] for c in oracle_ranked], gold, 10
                        )
                    },
                },
                "top10": {
                    "orig": [c["title"] for c in orig_ranked[:10]],
                    "self": [c["title"] for c in self_ranked[:10]],
                    "oracle": [c["title"] for c in oracle_ranked[:10]],
                },
            }
            results.append(row)
            print(
                f"  pool_gold={row['pool_gold_recall']:.2f} "
                f"orig={row['metrics']['orig']['Recall@10']:.2f} "
                f"self={row['metrics']['self']['Recall@10']:.2f} "
                f"oracle={row['metrics']['oracle']['Recall@10']:.2f}"
            )
            (out / "details.jsonl").open("a", encoding="utf-8").write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    n = len(results) or 1
    summary = {
        "status": "complete",
        "question_count": len(results),
        "unit": unit,
        "freeze": "MQ hop0 initial_candidate_ids ∪ all jev-scored ids from MQ run",
        "metrics": {
            "pool_gold_recall": sum(r["pool_gold_recall"] for r in results) / n,
            "orig_Recall@10": sum(r["metrics"]["orig"]["Recall@10"] for r in results) / n,
            "self_Recall@10": sum(r["metrics"]["self"]["Recall@10"] for r in results) / n,
            "oracle_Recall@10": sum(r["metrics"]["oracle"]["Recall@10"] for r in results)
            / n,
        },
        "per_question": [
            {
                "qid": r["dataset_sample_id"],
                "pool_gold_recall": r["pool_gold_recall"],
                "orig": r["metrics"]["orig"]["Recall@10"],
                "self": r["metrics"]["self"]["Recall@10"],
                "oracle": r["metrics"]["oracle"]["Recall@10"],
            }
            for r in results
        ],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[4]
    sag2 = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--sample-file",
        type=Path,
        default=repo / "dataset" / "musique_never11_for_mq.json",
    )
    p.add_argument(
        "--mq-run",
        type=Path,
        default=repo / "output" / "probe-never-multiquery-v2-27b" / "C6-MQ-H1-never11",
    )
    p.add_argument(
        "--dual-run",
        type=Path,
        default=repo
        / "output"
        / "probe-never-multiquery-v2-27b"
        / "C6-MQ-H1-dual-never11",
    )
    p.add_argument("--musique-json", type=Path, default=repo / "dataset" / "musique.json")
    p.add_argument("--dataset-dir", type=Path, default=repo / "dataset")
    p.add_argument("--cache-dir", type=Path, default=sag2 / ".cache" / "sag2")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "output"
        / "probe-never-multiquery-v2-27b"
        / "frozen-pool-three-arms",
    )
    p.add_argument("--evidence-model", default="qwen3.8-27b")
    p.add_argument("--evidence-batch-size", type=int, default=10)
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()
    # fresh details
    details = args.output_dir / "details.jsonl"
    if details.exists():
        details.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
