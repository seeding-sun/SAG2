"""OpenAI-compatible embedding client with batching, validation and replay."""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import aiohttp
import numpy as np

from model_api import RequestJournal


def embeddings_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    return endpoint if endpoint.endswith("/embeddings") else endpoint + "/embeddings"


class EmbeddingClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str,
        *,
        journal_dir: Path,
        batch_size: int = 32,
        concurrency: int = 2,
        normalize: bool = True,
        expected_dimension: int | None = None,
    ):
        if batch_size < 1 or concurrency < 1:
            raise ValueError("Embedding batch size and concurrency must be positive")
        self.session = session
        self.endpoint = embeddings_endpoint(base_url)
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.normalize = normalize
        self.expected_dimension = expected_dimension
        self.journal = RequestJournal(session, journal_dir)

    async def _embed_batch(self, texts: list[str], batch_number: int) -> tuple[np.ndarray, dict]:
        if not texts:
            raise ValueError("Embedding batch cannot be empty")
        payload = {"model": self.model, "input": texts}
        started = time.perf_counter()
        response = await self.journal.post(
            "embedding", self.endpoint, self.api_key, payload
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(texts)}, "
                f"received {len(rows) if isinstance(rows, list) else type(rows).__name__}"
            )
        indexed: dict[int, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Embedding response row must be an object")
            index = row.get("index")
            embedding = row.get("embedding")
            if type(index) is not int or not 0 <= index < len(texts):
                raise ValueError(f"Invalid embedding index: {index!r}")
            if index in indexed:
                raise ValueError(f"Duplicate embedding index: {index}")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Embedding row {index} has no vector")
            indexed[index] = embedding
        if set(indexed) != set(range(len(texts))):
            raise ValueError("Embedding indices do not cover the input batch")
        vectors = np.asarray([indexed[i] for i in range(len(texts))], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding response contains NaN or infinity")
        if self.expected_dimension is not None and vectors.shape[1] != self.expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dimension}, "
                f"received {vectors.shape[1]}"
            )
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Embedding response contains a zero or invalid norm")
            vectors /= norms
        usage = response.get("usage") or {}
        record = self.journal.records[-1]
        return vectors, {
            "batch": batch_number,
            "count": len(texts),
            "dimension": int(vectors.shape[1]),
            "model": response.get("model") or record.get("actual_model") or self.model,
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "reused": record.get("reused", False),
            "latency_ms": elapsed_ms,
        }

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: str = "passage",
    ) -> tuple[np.ndarray, dict]:
        if not texts:
            raise ValueError("At least one embedding input is required")
        if input_type not in {"passage", "query"}:
            raise ValueError("input_type must be 'passage' or 'query'")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        batches = [
            texts[start : start + self.batch_size]
            for start in range(0, len(texts), self.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run(number: int, batch: list[str]) -> tuple[np.ndarray, dict]:
            async with semaphore:
                return await self._embed_batch(batch, number)

        results = await asyncio.gather(
            *(run(number, batch) for number, batch in enumerate(batches))
        )
        vectors = np.concatenate([result[0] for result in results], axis=0)
        batches_detail = [result[1] for result in results]
        return vectors, {
            "input_type": input_type,
            "text_count": len(texts),
            "dimension": int(vectors.shape[1]),
            "requests": len(batches_detail),
            "input_tokens": sum(batch["input_tokens"] for batch in batches_detail),
            "models": sorted({batch["model"] for batch in batches_detail}),
            "batches": batches_detail,
        }

    async def embed_query(self, text: str) -> tuple[np.ndarray, dict]:
        vectors, usage = await self.embed([text], input_type="query")
        vector = vectors[0]
        if not math.isfinite(float(vector @ vector)):
            raise ValueError("Query embedding is invalid")
        return vector, usage
