import asyncio
import json

import numpy as np
from aiohttp import web

import run_hybrid_compare as runner
from analyze_hybrid_results import load_details, summarize
from sag2 import CorpusIndex, corpus_fingerprint


async def fake_embeddings(request):
    payload = await request.json()
    texts = payload["input"]
    data = [
        {"index": index, "embedding": [1.0, float(index + 1)]}
        for index, _ in enumerate(texts)
    ]
    return web.json_response(
        {
            "model": "fake-bge",
            "data": data,
            "usage": {"prompt_tokens": len(texts)},
        }
    )


async def fake_chat(request):
    payload = await request.json()
    content = json.loads(payload["messages"][1]["content"])
    scores = {
        key: (0.9 if "Anchor" in value["title"] else 0.2)
        for key, value in content["candidates"].items()
    }
    return web.json_response(
        {
            "model": "fake-qwen",
            "choices": [{"message": {"content": json.dumps({"scores": scores})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def prepare_fixture(root):
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    corpus = [
        {"title": "Anchor", "text": "anchor root"},
        {"title": "Bridge", "text": "anchor target"},
        {"title": "Distractor", "text": "unrelated"},
    ]
    questions = [
        {
            "id": "2hop__1_2",
            "question": "anchor target",
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Anchor",
                    "paragraph_text": "anchor root",
                    "is_supporting": True,
                },
                {
                    "idx": 1,
                    "title": "Bridge",
                    "paragraph_text": "anchor target",
                    "is_supporting": True,
                },
            ],
        }
    ]
    write_json(dataset_dir / "musique_corpus.json", corpus)
    write_json(dataset_dir / "musique.json", questions)
    sample = dataset_dir / "sample.json"
    write_json(sample, {"ids": ["2hop__1_2"]})
    cache_dir = root / "cache"
    cache_dir.mkdir()
    index_path = cache_dir / "musique.sqlite"
    corpus_hash = corpus_fingerprint(dataset_dir / "musique_corpus.json")
    index = CorpusIndex(index_path, corpus, corpus_hash, index_workers=1)
    ids = [
        row["id"]
        for row in index.db.execute("SELECT id FROM chunks ORDER BY id")
    ]
    index.close()
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    stem = cache_dir / "musique-bge-m3"
    with stem.with_suffix(".npy").open("wb") as stream:
        np.save(stream, vectors)
    write_json(stem.with_suffix(".ids.json"), ids)
    write_json(
        stem.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "dataset": "musique",
            "corpus_sha256": corpus_hash,
            "index_signature": "fixture",
            "chunk_policy": "fixture",
            "chunk_chars": 2200,
            "retrieval_unit": "chunk",
            "embedding_model": "fake-bge",
            "dimension": 2,
            "count": len(ids),
            "normalized": True,
        },
    )
    return dataset_dir, sample, cache_dir


def test_runner_executes_both_arms_and_paired_analysis(tmp_path):
    dataset_dir, sample, cache_dir = prepare_fixture(tmp_path)

    async def exercise():
        app = web.Application()
        app.router.add_post("/v1/embeddings", fake_embeddings)
        app.router.add_post("/v1/chat/completions", fake_chat)
        runner_http = web.AppRunner(app)
        await runner_http.setup()
        site = web.TCPSite(runner_http, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets
        port = sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            for mode, name in (("bm25", "A"), ("hybrid", "B")):
                args = runner.build_parser().parse_args(
                    [
                        "--retrieval-mode",
                        mode,
                        "--dataset-dir",
                        str(dataset_dir),
                        "--sample-file",
                        str(sample),
                        "--output-dir",
                        str(tmp_path / name),
                        "--cache-dir",
                        str(cache_dir),
                        "--max-hops",
                        "0",
                        "--concurrency",
                        "1",
                        "--index-workers",
                        "1",
                        "--embedding-base-url",
                        base_url,
                        "--embedding-model",
                        "fake-bge",
                        "--embedding-dimension",
                        "2",
                        "--evidence-base-url",
                        base_url,
                        "--evidence-model",
                        "fake-qwen",
                        "--evidence-batch-size",
                        "10",
                        "--evidence-concurrency",
                        "1",
                    ]
                )
                await runner.run(args)
        finally:
            await runner_http.cleanup()

    asyncio.run(exercise())
    a_rows = load_details(tmp_path / "A")
    b_rows = load_details(tmp_path / "B")
    result = summarize(a_rows, b_rows, samples=20, seed=1)
    assert result["question_count"] == 1
    assert result["cost"]["a_judgments"] == 2
    assert result["cost"]["b_judgments"] == 3
