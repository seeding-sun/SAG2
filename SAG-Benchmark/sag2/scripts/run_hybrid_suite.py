#!/usr/bin/env python3
"""Prepare and run the A/B/C hybrid experiment suite on a server."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_SPECS = {
    "bm25": ("A", "bm25"),
    "hybrid": ("B", "hybrid"),
    "dense": ("C", "dense"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "sag2")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ARM_SPECS),
        default=["bm25", "dense", "hybrid"],
    )
    parser.add_argument("--max-hops", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed-k", type=int, default=10)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--evidence-concurrency", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-concurrency", type=int, default=2)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("RUN:", " ".join(command), flush=True)
    if dry_run:
        return
    environment = dict(os.environ)
    environment.setdefault("PYTHONUTF8", "1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def prepare(args: argparse.Namespace) -> None:
    index_path = args.cache_dir / f"{args.dataset}.sqlite"
    if not index_path.is_file():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_sag2_benchmark.py"),
            "--datasets",
            args.dataset,
            "--dataset-dir",
            str(args.dataset_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--index-only",
            "--index-workers",
            str(args.index_workers),
        ]
        run_command(command, dry_run=args.dry_run)
    needs_vectors = any(arm in {"dense", "hybrid"} for arm in args.arms)
    vector_path = args.cache_dir / f"{args.dataset}-bge-m3.npy"
    if needs_vectors and not vector_path.is_file():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_passage_embeddings.py"),
            "--dataset",
            args.dataset,
            "--dataset-dir",
            str(args.dataset_dir),
            "--index",
            str(index_path),
            "--cache-dir",
            str(args.cache_dir),
            "--batch-size",
            str(args.embedding_batch_size),
            "--concurrency",
            str(args.embedding_concurrency),
            "--dimension",
            str(args.embedding_dimension),
            "--env-file",
            str(args.env_file),
        ]
        run_command(command, dry_run=args.dry_run)


def runner_command(args: argparse.Namespace, arm: str) -> tuple[str, list[str]]:
    letter, mode = ARM_SPECS[arm]
    tag = f"{letter}1-H{args.max_hops}"
    output = args.output_root / tag
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_hybrid_compare.py"),
        "--retrieval-mode",
        mode,
        "--dataset",
        args.dataset,
        "--dataset-dir",
        str(args.dataset_dir),
        "--sample-file",
        str(args.sample_file),
        "--output-dir",
        str(output),
        "--cache-dir",
        str(args.cache_dir),
        "--max-hops",
        str(args.max_hops),
        "--candidate-k",
        str(args.candidate_k),
        "--seed-k",
        str(args.seed_k),
        "--index-workers",
        str(args.index_workers),
        "--concurrency",
        str(args.concurrency),
        "--evidence-concurrency",
        str(args.evidence_concurrency),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        "--embedding-concurrency",
        str(args.embedding_concurrency),
        "--embedding-dimension",
        str(args.embedding_dimension),
        "--env-file",
        str(args.env_file),
    ]
    if args.resume:
        command.append("--resume")
    return tag, command


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_prepare:
        prepare(args)
    if args.prepare_only:
        return
    for arm in args.arms:
        tag, command = runner_command(args, arm)
        print(f"ARM {arm} -> {tag}", flush=True)
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
