"""Dense initial retrieval with LLM query decomposition + RRF fusion."""

from __future__ import annotations

import json
import re
from collections import defaultdict

import aiohttp

from sag2 import CorpusIndex, compact_json
from vector_retriever import VectorRetriever


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
    retries: int = 3,
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
            return json.loads(_extract_json_object(str(content)))
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(f"llm_json failed: {last_error}")


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
        row["score"] = score
        out.append(row)
    return out


class MultiQueryDenseRetriever:
    """Hop-0 dense retrieval over original query + LLM sub-questions, RRF-fused."""

    name = "dense-mq"

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        *,
        session: aiohttp.ClientSession,
        endpoint: str,
        model: str,
        api_key: str,
        per_query_k: int = 80,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.session = session
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.per_query_k = per_query_k
        self.rrf_k = rrf_k

    async def decompose(self, question: str) -> list[str]:
        obj = await llm_json(
            self.session,
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            system=DECOMPOSE_SYSTEM,
            user=DECOMPOSE_USER.format(question=question),
        )
        return [str(s).strip() for s in (obj.get("sub_questions") or []) if str(s).strip()]

    async def followup_queries(self, question: str, seeds: list[dict]) -> list[str]:
        seed_blob = "\n".join(
            f"- {s.get('title', '')}: {(s.get('content') or s.get('passage') or '')[:400]}"
            for s in seeds
        ) or "(none)"
        obj = await llm_json(
            self.session,
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            system=FOLLOWUP_SYSTEM,
            user=FOLLOWUP_USER.format(question=question, seeds=seed_blob),
        )
        return [str(q).strip() for q in (obj.get("queries") or []) if str(q).strip()]

    async def search(
        self,
        index: CorpusIndex,
        query: str,
        *,
        excluded: set[int],
        limit: int,
    ) -> tuple[list[dict], dict]:
        sub_qs = await self.decompose(query)
        queries = [query, *sub_qs]
        hit_lists: list[list[dict]] = []
        usages = []
        for q in queries:
            hits, usage = await self.vector_retriever.search(
                index, q, limit=self.per_query_k, excluded=excluded
            )
            hit_lists.append(hits)
            usages.append({"query": q, **usage})
        fused = rrf_merge(hit_lists, rrf_k=self.rrf_k, limit=limit)
        for rank, row in enumerate(fused, start=1):
            row["initial_rank"] = rank
            row["retrieval_route"] = self.name
        return fused, {
            "retriever": self.name,
            "sub_questions": sub_qs,
            "queries": queries,
            "per_query_k": self.per_query_k,
            "rrf_k": self.rrf_k,
            "candidate_count": len(fused),
            "routes": usages,
        }


async def dense_search_queries(
    vector_retriever: VectorRetriever,
    index: CorpusIndex,
    queries: list[str],
    *,
    excluded: set[int],
    per_query_k: int,
    limit: int,
    rrf_k: int = 60,
) -> list[dict]:
    hit_lists = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        hits, _ = await vector_retriever.search(
            index, q, limit=per_query_k, excluded=excluded
        )
        hit_lists.append(hits)
    return rrf_merge(hit_lists, rrf_k=rrf_k, limit=limit) if hit_lists else []
