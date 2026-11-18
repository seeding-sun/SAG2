"""Hop bridge-question scoring: structured {E}+{R} anchors with admission gates.

Spec (doc §5.8):
  - Static decompose sub-questions → retrieval only (MQ).
  - Dynamic bridge sub-questions → scoring only, template ``{E} {R_phrase}?``.
  - Dual-path grade merge: score = max(s_orig, s_bridge)  (not element-wise emb max).
  - Bridge scoring enabled only if entity confidence + non-ambiguous + single-valued R.
  - Original-question path never disabled.
"""

from __future__ import annotations

import re
from typing import Any

import aiohttp

from multi_query_retriever import (
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    dense_search_queries,
    llm_json,
)
from sag2 import GRADE_RUBRIC


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

SLOT_CONFLICT_SYSTEM = (
    "You list distinct short answers for one entity-relation from passages. "
    "Return only one JSON object."
)
SLOT_CONFLICT_USER = (
    "Subject entity E: {entity}\n"
    "Relation R: {relation}\n"
    "Passages:\n{passages}\n\n"
    "List every DISTINCT short answer value that a passage claims for E under R. "
    "Separate corporate/administrative headquarters from factories, distilleries, "
    "branches, or plants when relation is headquarters. "
    'Return {{"answers": ["...", "..."]}} only. Use [] if none.'
)

VERIFY_SYSTEM = (
    "You are a strict relation verifier. Return only one token."
)
VERIFY_USER = (
    "Subject entity E: {entity}\n"
    "Target relation R: {relation}\n"
    "Sentence S title: {title}\n"
    "Sentence S text: {text}\n\n"
    "Judge whether S directly states the PRIMARY official fact that E has relation R. "
    "Distinguish primary official facts from secondary/affiliated/derivative sites.\n"
    "For headquarters: valid=corporate/administrative head office; "
    "invalid=factory/distillery/branch/warehouse/subsidiary/production site; "
    "uncertain=cannot tell.\n"
    "For birth_place: valid=place of birth; invalid=hometown/residence/worked in; "
    "uncertain=ambiguous.\n"
    "Output exactly one token: valid OR invalid OR uncertain"
)

# Relations that run per-passage verifier after layer-1 gates.
VERIFY_RELATIONS = {"headquarters", "birth_place"}

# Relations that require slot-value semantic check before opening the bridge.
SEMANTIC_RELATIONS = {"headquarters", "birth_place"}

SLOT_SEMANTIC_SYSTEM = (
    "You check whether a candidate slot value is a PRIMARY fact for a relation. "
    "Return only one JSON object."
)
SLOT_SEMANTIC_USER = (
    "Subject entity E: {entity}\n"
    "Relation R: {relation}\n"
    "Candidate value V: {value}\n"
    "Passages:\n{passages}\n\n"
    "Decide if V is supported as the PRIMARY official value of R for E.\n"
    "For headquarters: accept only corporate/administrative head office wording "
    "(e.g. headquartered in, headquarters in, corporate HQ). "
    "Reject factory/distillery/plant/branch/warehouse/subsidiary/production-site "
    "wording even if V appears in the text.\n"
    "For birth_place: accept only born/birthplace wording; reject hometown/"
    "residence/worked-in/album-blurb geography.\n"
    "Single-passage strong primary wording is enough to accept "
    "(do NOT require multiple sources).\n"
    "If passages give conflicting primary cities/places, reject.\n"
    'Return {{"verdict": "accept"|"reject", "strength": "strong"|"weak"|"conflict"|"none", '
    '"reason": "short"}} only.'
)

# Single-valued bridge relations allowed into scoring max.
RELATION_PHRASES: dict[str, str] = {
    "birth_place": "place of birth",
    "headquarters": "headquarters location",
    "capital_of": "capital city",
    "manufacturer": "manufacturer",
    "performer": "performer",
    "state_of": "home state",
    "country_of": "country",
}

# Patterns → relation id (first match wins). Order matters.
_SINGLE_PATTERNS: list[tuple[str, str]] = [
    (r"birthplace|place of birth|(?:was |be )?born", "birth_place"),
    (r"headquarters|headquartered|\bhq\b", "headquarters"),
    (r"capital of|state capital|capital city|largest city and capital", "capital_of"),
    (r"manufacturer|manufactured|produced by|maker of", "manufacturer"),
    # state_of before performer: "state … performer of X from" must not hit performer.
    (r"(?:what|which) state\b|home state|state is .* from|from\?$", "state_of"),
    (r"what(?: is)? the country|which country|country where", "country_of"),
    (r"who (?:is|was) the performer|performer of the (?:song|album)|sang |artist of", "performer"),
]

# Ambiguous / multi-valued → never enable bridge scoring.
_AMBIGUOUS_REL_PATTERNS: list[str] = [
    r"highest|top scorer|most goals|all-?time",
    r"\bmember of\b|currently a member|which team is",
    r"shares? a border|bordering|adjacent",
    r"most (?:seats|tourists)",
    r"first black mayor",
    r"indy car|won the .*race",
]

# Seed ambiguity locklist (extend as needed).
AMBIGUOUS_ENTITIES = {
    "urim",
    "the world",
}


def fill_placeholders(template: str, answers: dict[int, str]) -> str:
    out = template
    for idx, ans in answers.items():
        if not ans:
            continue
        out = out.replace(f"{{{{{idx}}}}}", ans)
        out = out.replace(f"{{{idx}}}", ans)
    return out


def has_open_placeholder(text: str) -> bool:
    return bool(re.search(r"\{\{?\d+\}?\}", text or ""))


def snap_blend(hop_grade: int, orig_grade: int, alpha: float = 0.7) -> int:
    raw = alpha * hop_grade + (1.0 - alpha) * orig_grade
    allowed = sorted(GRADE_RUBRIC)
    return min(allowed, key=lambda g: (abs(g - raw), -g))


def passages_blob(passages: list[dict], *, n: int = 8) -> str:
    parts = []
    for p in passages[:n]:
        title = p.get("title") or ""
        content = (p.get("content") or p.get("passage") or "")[:500]
        parts.append(f"- {title}: {content}")
    return "\n".join(parts) if parts else "(none)"


def classify_relation(text: str) -> tuple[str | None, bool]:
    """Return (relation_id or None, is_ambiguous_relation)."""
    q = (text or "").lower()
    for pat in _AMBIGUOUS_REL_PATTERNS:
        if re.search(pat, q):
            return None, True
    for pat, rid in _SINGLE_PATTERNS:
        if re.search(pat, q):
            return rid, False
    return None, False


def build_bridge_question(entity: str, relation_id: str) -> str:
    phrase = RELATION_PHRASES.get(relation_id, relation_id.replace("_", " "))
    ent = entity.strip()
    # Fixed templates — no free rewrite.
    if relation_id == "birth_place":
        return f"What is the place of birth of {ent}?"
    if relation_id == "headquarters":
        return f"Where is the headquarters of {ent}?"
    if relation_id == "capital_of":
        return f"What is the capital of {ent}?"
    if relation_id == "manufacturer":
        return f"Who is the manufacturer of {ent}?"
    if relation_id == "performer":
        return f"Who is the performer of {ent}?"
    if relation_id == "state_of":
        return f"What state is {ent} from?"
    if relation_id == "country_of":
        return f"Which country is {ent} located in?"
    return f"What is the {phrase} of {ent}?"


def verify_queries_for_relation(entity: str, relation_id: str) -> list[str]:
    ent = (entity or "").strip()
    if not ent or not relation_id:
        return []
    if relation_id == "birth_place":
        return [f"{ent} place of birth", f"{ent} born in", f"Where was {ent} born?"]
    if relation_id == "headquarters":
        # Prefer corporate HQ wording to reduce distillery/plant city confounds.
        return [
            f"{ent} corporate headquarters",
            f"{ent} company headquarters",
            f"Where is {ent} headquartered?",
            f"{ent} headquarters city",
        ]
    if relation_id == "capital_of":
        return [f"capital of {ent}", f"What is the capital of {ent}?"]
    if relation_id == "manufacturer":
        return [f"{ent} manufacturer", f"{ent} produced by"]
    if relation_id == "performer":
        return [f"{ent} performer", f"Who performed {ent}?"]
    if relation_id == "state_of":
        return [f"{ent} home state", f"What state is {ent} from?"]
    if relation_id == "country_of":
        return [f"{ent} country", f"Which country is {ent} in?"]
    return [f"{ent} {RELATION_PHRASES.get(relation_id, '')}"]


def entity_ambiguous(name: str) -> bool:
    key = (name or "").strip().lower()
    if not key:
        return True
    if key in AMBIGUOUS_ENTITIES:
        return True
    # Very short tokens often collide in MuSiQue bridges.
    if len(key) <= 3:
        return True
    return False


def _norm_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class RelationVerifier:
    """Per-passage primary-vs-affiliated fact gate for high-risk relations."""

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        endpoint: str,
        model: str,
        api_key: str,
        uncertain_scale: float = 0.2,
        top_n: int = 20,
    ):
        self.session = session
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.uncertain_scale = uncertain_scale
        self.top_n = top_n
        self.cache: dict[tuple[str, str, int], str] = {}

    async def judge(
        self, *, entity: str, relation: str, cand_id: int, title: str, text: str
    ) -> str:
        key = (entity.lower(), relation, int(cand_id))
        if key in self.cache:
            return self.cache[key]
        label = await llm_verify_token(
            self.session,
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            user=VERIFY_USER.format(
                entity=entity,
                relation=relation,
                title=(title or "")[:200],
                text=(text or "")[:600],
            ),
        )
        self.cache[key] = label if label in {"valid", "invalid", "uncertain"} else "uncertain"
        return self.cache[key]


async def llm_verify_token(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    user: str,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    if "deepseek" in model.lower():
        payload["reasoning_effort"] = "none"
        payload["thinking"] = False
    async with session.post(
        endpoint, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        body = await resp.json(content_type=None)
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {body}")
    message = body["choices"][0]["message"]
    content = str(message.get("content") or message.get("reasoning_content") or "").strip().lower()
    for tok in ("valid", "invalid", "uncertain"):
        if content == tok or content.startswith(tok) or tok in content.split():
            return tok
    m = re.search(r"valid|invalid|uncertain", content)
    return m.group(0) if m else "uncertain"


class HopAnchorPlanner:
    """Decompose for retrieval context; emit gated structured bridge anchors for scoring."""

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        endpoint: str,
        model: str,
        api_key: str,
        vector_retriever: Any | None = None,
        per_query_k: int = 40,
        verify: bool = True,
        gate: bool = True,
        entity_tau: float = 0.6,
        slot_conflict: bool = True,
        passage_verify: bool = False,
        slot_semantic: bool = False,
        uncertain_scale: float = 0.2,
        verify_top_n: int = 20,
    ):
        self.session = session
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.vector_retriever = vector_retriever
        self.per_query_k = per_query_k
        self.verify = verify
        self.gate = gate
        self.entity_tau = entity_tau
        self.slot_conflict = slot_conflict
        self.passage_verify = passage_verify
        self.slot_semantic = slot_semantic
        self.uncertain_scale = uncertain_scale
        self.verify_top_n = verify_top_n
        self.templates: list[str] = []
        self.answers: dict[int, str] = {}
        # step -> {entity, score, ambiguous, relation_for_next, conflict}
        self.answer_meta: dict[int, dict] = {}
        self.current_hop = 0
        self.trace: list[dict] = []
        self.last_gate: dict = {"enabled": False, "reason": "init", "anchors": []}
        self.verifier = RelationVerifier(
            session=session,
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            uncertain_scale=uncertain_scale,
            top_n=verify_top_n,
        )

    async def ensure_decomposed(self, question: str) -> list[str]:
        if self.templates:
            return self.templates
        if not (question or "").strip():
            return []
        self.templates = [
            str(s).strip()
            for s in (
                await llm_json(
                    self.session,
                    endpoint=self.endpoint,
                    model=self.model,
                    api_key=self.api_key,
                    system=DECOMPOSE_SYSTEM,
                    user=DECOMPOSE_USER.format(question=question),
                )
            ).get("sub_questions")
            or []
            if str(s).strip()
        ]
        self.trace.append({"stage": "decompose", "sub_questions": list(self.templates)})
        return self.templates

    def set_hop(self, hop: int) -> None:
        self.current_hop = hop

    def filled_all(self) -> list[str]:
        out = []
        for tmpl in self.templates:
            filled = fill_placeholders(tmpl, self.answers)
            if not has_open_placeholder(filled):
                out.append(filled)
        return out

    def scoring_anchors(self) -> list[str]:
        """Return structured bridge questions if gates pass; else [].

        Hop0: orig-only (no prior entity).
        Hop k>=1: entity from answers[k] + single-valued R from template[k].

        When ``gate=False`` (legacy dual): fall back to filled free-form templates
        up to the current hop (ungated) for A/B comparison.
        """
        self.last_gate = {
            "enabled": False,
            "reason": "no_bridge",
            "anchors": [],
            "hop": self.current_hop,
            "gate": self.gate,
        }
        if not self.gate:
            # Legacy permissive path (pre-§5.8).
            if not self.templates:
                return []
            idx = min(self.current_hop, len(self.templates) - 1)
            anchors: list[str] = []
            seen: set[str] = set()
            for i, tmpl in enumerate(self.templates):
                filled = fill_placeholders(tmpl, self.answers)
                if has_open_placeholder(filled):
                    continue
                if i <= idx and filled not in seen:
                    seen.add(filled)
                    anchors.append(filled)
            self.last_gate = {
                "enabled": bool(anchors),
                "reason": "legacy_ungated",
                "anchors": anchors,
                "hop": self.current_hop,
                "gate": False,
            }
            return anchors

        if self.current_hop <= 0:
            self.last_gate["reason"] = "hop0_orig_only"
            return []
        if not self.templates:
            self.last_gate["reason"] = "no_templates"
            return []

        prev_step = self.current_hop
        entity = (self.answers.get(prev_step) or "").strip()
        meta = self.answer_meta.get(prev_step) or {}
        if not entity:
            self.last_gate["reason"] = "no_entity"
            return []

        next_idx = self.current_hop
        if next_idx >= len(self.templates):
            self.last_gate["reason"] = "no_next_template"
            return []
        next_tmpl = fill_placeholders(self.templates[next_idx], self.answers)
        rel_id, rel_ambiguous = classify_relation(next_tmpl)

        reasons_fail = []
        score = float(meta.get("score") or 0.0)
        if score < self.entity_tau:
            reasons_fail.append(f"entity_score<{self.entity_tau} ({score:.2f})")
        if meta.get("ambiguous") or entity_ambiguous(entity) or meta.get("conflict"):
            reasons_fail.append("entity_ambiguous_or_conflict")
        if rel_ambiguous:
            reasons_fail.append("relation_ambiguous")
        if rel_id is None:
            reasons_fail.append("relation_not_single_valued")
        # High-risk R: require semantic accept on the slot (entity or next value).
        if self.slot_semantic and rel_id in SEMANTIC_RELATIONS:
            nxt_meta = self.answer_meta.get(self.current_hop + 1) or {}
            sem = meta.get("slot_semantic") or nxt_meta.get("slot_semantic")
            if sem == "reject":
                reasons_fail.append("slot_semantic_reject")
            elif sem != "accept":
                reasons_fail.append("slot_semantic_missing")

        if reasons_fail:
            self.last_gate = {
                "enabled": False,
                "reason": ";".join(reasons_fail),
                "anchors": [],
                "entity": entity,
                "relation": rel_id,
                "entity_score": meta.get("score"),
                "hop": self.current_hop,
                "gate": True,
            }
            self.trace.append({"stage": "gate_reject", **self.last_gate})
            return []

        bridge = build_bridge_question(entity, rel_id)
        self.last_gate = {
            "enabled": True,
            "reason": "ok",
            "anchors": [bridge],
            "entity": entity,
            "relation": rel_id,
            "entity_score": meta.get("score"),
            "hop": self.current_hop,
            "gate": True,
        }
        self.trace.append({"stage": "gate_accept", **self.last_gate})
        return [bridge]

    async def _extract(self, sub_question: str, passages: list[dict]) -> str:
        obj = await llm_json(
            self.session,
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            system=EXTRACT_SYSTEM,
            user=EXTRACT_USER.format(
                sub_question=sub_question, passages=passages_blob(passages)
            ),
        )
        return str(obj.get("answer") or "").strip()

    async def _list_slot_answers(
        self, *, entity: str, relation: str, passages: list[dict]
    ) -> list[str]:
        """Collect distinct slot values for E+R; used for multi-answer conflict."""
        if not entity or not relation or not passages:
            return []
        obj = await llm_json(
            self.session,
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            system=SLOT_CONFLICT_SYSTEM,
            user=SLOT_CONFLICT_USER.format(
                entity=entity,
                relation=relation,
                passages=passages_blob(passages, n=12),
            ),
        )
        raw = obj.get("answers") if isinstance(obj, dict) else []
        out: list[str] = []
        seen: set[str] = set()
        for a in raw or []:
            s = str(a).strip()
            n = _norm_answer(s)
            if not s or n in seen:
                continue
            # soft dedupe: "Frankfort" vs "Frankfort, Kentucky"
            if any(n in x or x in n for x in seen):
                continue
            seen.add(n)
            out.append(s)
        return out

    async def _slot_semantic_check(
        self, *, entity: str, relation: str, value: str, passages: list[dict]
    ) -> dict:
        """Return {verdict, strength, reason} for primary-fact support of V."""
        if not (entity and relation and value and passages):
            return {"verdict": "reject", "strength": "none", "reason": "missing_inputs"}
        obj = await llm_json(
            self.session,
            endpoint=self.endpoint,
            model=self.model,
            api_key=self.api_key,
            system=SLOT_SEMANTIC_SYSTEM,
            user=SLOT_SEMANTIC_USER.format(
                entity=entity,
                relation=relation,
                value=value,
                passages=passages_blob(passages, n=12),
            ),
        )
        if not isinstance(obj, dict):
            return {"verdict": "reject", "strength": "none", "reason": "bad_json"}
        verdict = str(obj.get("verdict") or "").strip().lower()
        if verdict not in {"accept", "reject"}:
            # Map strength-only replies.
            strength = str(obj.get("strength") or "").strip().lower()
            verdict = "accept" if strength == "strong" else "reject"
        strength = str(obj.get("strength") or "").strip().lower() or "none"
        reason = str(obj.get("reason") or "").strip()
        return {"verdict": verdict, "strength": strength, "reason": reason}

    async def _apply_slot_semantic(
        self,
        *,
        entity: str,
        relation: str,
        value: str,
        passages: list[dict],
        entity_step: int,
        value_step: int,
        hop: int,
    ) -> bool:
        """Run semantic check; stamp metas. Returns True if accept."""
        if not self.slot_semantic or relation not in SEMANTIC_RELATIONS:
            return True
        result = await self._slot_semantic_check(
            entity=entity, relation=relation, value=value, passages=passages
        )
        accept = result.get("verdict") == "accept"
        label = "accept" if accept else "reject"
        for step in {entity_step, value_step}:
            meta = self.answer_meta.get(step)
            if meta is None:
                continue
            meta["slot_semantic"] = label
            meta["slot_semantic_detail"] = result
            if not accept:
                meta["conflict"] = True
        self.trace.append(
            {
                "stage": "slot_semantic",
                "hop": hop,
                "entity": entity,
                "relation": relation,
                "value": value,
                "verdict": label,
                "detail": result,
            }
        )
        return accept

    @staticmethod
    def _entity_support_score(entity: str, passages: list[dict]) -> float:
        """Confidence proxy: max jev_score among passages mentioning the entity.

        Title hits get a floor of 0.7 — under gated hop0, bridge docs often stay at
        grade 0 / low jev even when retrieval clearly surfaced the right title.
        """
        if not entity:
            return 0.0
        key = entity.lower()
        best = 0.0
        for p in passages or []:
            title = (p.get("title") or "").lower()
            content = (p.get("content") or p.get("passage") or "").lower()
            if key in title:
                best = max(best, max(float(p.get("jev_score") or 0.0), 0.7))
            elif key in content:
                best = max(best, float(p.get("jev_score") or 0.0))
        if best > 0:
            return best
        return max((float(p.get("jev_score") or 0.0) for p in (passages or [])), default=0.0)

    def _enrich_passages(self, index: Any, passages: list[dict]) -> list[dict]:
        """Ensure title/content are present (pull from chunk index when missing)."""
        out = []
        for p in passages or []:
            row = dict(p)
            need = not (row.get("content") or row.get("passage"))
            cid = row.get("id")
            if need and cid is not None and index is not None:
                try:
                    got = index.db.execute(
                        "SELECT title, content FROM chunks WHERE id = ?", (int(cid),)
                    ).fetchone()
                    if got:
                        row["title"] = row.get("title") or got["title"] or ""
                        row["content"] = got["content"] or ""
                except Exception:  # noqa: BLE001
                    pass
            out.append(row)
        return out

    async def update_after_hop(
        self,
        *,
        index: Any,
        frontier: list[dict],
        hop: int,
    ) -> None:
        """Extract answer for sub-question (hop+1); optional verify; record gate meta."""
        await self.ensure_decomposed("")
        if not self.templates:
            return
        step = hop + 1
        if step > len(self.templates):
            return
        template = self.templates[step - 1]
        sub_q = fill_placeholders(template, self.answers)
        if has_open_placeholder(sub_q):
            self.trace.append(
                {
                    "stage": "extract_skip",
                    "hop": hop,
                    "step": step,
                    "reason": "open_placeholder",
                }
            )
            return

        passages = self._enrich_passages(index, frontier or [])
        answer = await self._extract(sub_q, passages)
        # Title cue fallback: "Oh Yeah (Charles Mingus album)" → Charles Mingus
        if not answer:
            for p in passages:
                title = p.get("title") or ""
                m = re.search(r"\(([^)]+?)\s+album\)", title, re.I)
                if m and re.search(r"performer|sang|artist|album", sub_q, re.I):
                    answer = m.group(1).strip()
                    break
        rel_id, rel_ambiguous = classify_relation(sub_q)
        verify_used: list[str] = []
        conflict = False
        answer_verify = ""
        merged = list(passages)

        # Verify only for single-valued relations with a known entity context.
        entity_for_verify = ""
        if step == 1 and answer:
            entity_for_verify = answer
            # Pre-verify *next* relation if single-valued.
            if step < len(self.templates):
                next_filled = fill_placeholders(
                    self.templates[step], {**self.answers, step: answer}
                )
                next_rel, next_amb = classify_relation(next_filled)
                if next_rel and not next_amb:
                    rel_id = next_rel
        elif step > 1:
            entity_for_verify = (self.answers.get(step - 1) or "").strip()

        if (
            self.verify
            and self.vector_retriever is not None
            and entity_for_verify
            and rel_id
            and not rel_ambiguous
        ):
            vqs = verify_queries_for_relation(entity_for_verify, rel_id)
            if vqs:
                dense_hits = await dense_search_queries(
                    self.vector_retriever,
                    index,
                    vqs,
                    excluded=set(),
                    per_query_k=self.per_query_k,
                    limit=30,
                )
                verify_used = vqs
                merged = dense_hits + passages
                # If verifying next relation after step1 person/org extract:
                if step == 1 and answer and step < len(self.templates):
                    next_q = fill_placeholders(
                        self.templates[step], {**self.answers, step: answer}
                    )
                    next_rel, next_amb = classify_relation(next_q)
                    if next_rel and not next_amb:
                        next_ans = await self._extract(next_q, merged)
                        if next_ans:
                            self.answers[step + 1] = next_ans
                            self.answer_meta[step + 1] = {
                                "entity": next_ans,
                                "score": self._entity_support_score(next_ans, merged),
                                "ambiguous": entity_ambiguous(next_ans),
                                "conflict": False,
                                "relation": next_rel,
                                "source": "verify_next",
                            }
                            # J: semantic check on the pre-filled next slot (e.g. HQ city).
                            ok = await self._apply_slot_semantic(
                                entity=answer,
                                relation=next_rel,
                                value=next_ans,
                                passages=merged,
                                entity_step=step,
                                value_step=step + 1,
                                hop=hop,
                            )
                            if not ok:
                                conflict = True
                answer_verify = await self._extract(sub_q, merged)
                if answer_verify and answer and answer_verify.lower() != answer.lower():
                    conflict = True
                    # Do NOT silently prefer one side — mark conflict and keep frontier
                    # answer; bridge scoring will be disabled by the gate.
                elif answer_verify:
                    answer = answer_verify

                # Layer-2: same E+R multi-answer conflict over merged evidence.
                if self.slot_conflict and entity_for_verify and rel_id:
                    competing = await self._list_slot_answers(
                        entity=entity_for_verify,
                        relation=rel_id,
                        passages=merged,
                    )
                    if len(competing) >= 2:
                        conflict = True
                        nxt = step + 1
                        if nxt in self.answer_meta:
                            self.answer_meta[nxt]["conflict"] = True
                        self.trace.append(
                            {
                                "stage": "slot_conflict",
                                "hop": hop,
                                "step": step,
                                "entity": entity_for_verify,
                                "relation": rel_id,
                                "answers": competing,
                            }
                        )

        prior_sem = {
            k: v
            for k, v in (self.answer_meta.get(step) or {}).items()
            if k in {"slot_semantic", "slot_semantic_detail"}
        }
        support = self._entity_support_score(answer, passages)
        if answer:
            self.answers[step] = answer
        self.answer_meta[step] = {
            "entity": answer,
            "score": support,
            "ambiguous": entity_ambiguous(answer) or rel_ambiguous,
            "conflict": conflict,
            "relation": rel_id,
            "source": "extract",
            **prior_sem,
        }
        # J: semantic-check current high-risk slot (step>1 only; step1 uses verify_next path).
        if (
            self.slot_semantic
            and step > 1
            and answer
            and entity_for_verify
            and rel_id in SEMANTIC_RELATIONS
            and self.answer_meta[step].get("slot_semantic") is None
        ):
            ok = await self._apply_slot_semantic(
                entity=entity_for_verify,
                relation=rel_id,
                value=answer,
                passages=merged,
                entity_step=step - 1,
                value_step=step,
                hop=hop,
            )
            if not ok:
                self.answer_meta[step]["conflict"] = True
        self.trace.append(
            {
                "stage": "extract",
                "hop": hop,
                "step": step,
                "sub_question": sub_q,
                "answer": answer,
                "answer_verify": answer_verify,
                "verify_queries": verify_used,
                "meta": dict(self.answer_meta[step]),
                "answers_so_far": dict(self.answers),
            }
        )


class DualAnchorScorer:
    """Dual-path evidence grades: max(s_orig, s_bridge) when planner gates pass.

    Optional per-passage verifier (high-risk R only) gates s2:
      valid → keep s2; invalid → s2 dropped; uncertain → s2 scaled down.
    """

    def __init__(
        self,
        inner: Any,
        planner: HopAnchorPlanner,
        *,
        mode: str = "max",
        alpha: float = 0.7,
    ):
        self.inner = inner
        self.planner = planner
        self.mode = mode
        self.alpha = alpha

    async def score(
        self,
        query: str,
        candidates: list[dict],
        evidence: list[dict],
        *,
        context: dict | None = None,
    ) -> tuple[list[dict], dict]:
        orig_rows, orig_usage = await self.inner.score(
            query, candidates, evidence, context=context
        )
        anchors = self.planner.scoring_anchors()
        gate_info = dict(getattr(self.planner, "last_gate", {}) or {})
        if not anchors or not candidates:
            for row in orig_rows:
                row["jev_grade_orig"] = int(
                    row.get("jev_grade", round(float(row["jev_score"]) * 10))
                )
                row["jev_grade_hop"] = None
            return orig_rows, {
                **orig_usage,
                "dual_anchor": {
                    "mode": self.mode,
                    "anchors": [],
                    "applied": False,
                    "gate": gate_info,
                },
            }

        best_hop: dict[int, int] = {int(c["id"]): 0 for c in candidates}
        hop_usage_sum = {
            "requests": 0,
            "judgments": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "imputed_judgments": 0,
            "batches": [],
        }
        for anchor in anchors:
            hop_rows, hop_usage = await self.inner.score(
                anchor, candidates, evidence, context=context
            )
            for key in (
                "requests",
                "judgments",
                "input_tokens",
                "output_tokens",
                "imputed_judgments",
            ):
                hop_usage_sum[key] += hop_usage.get(key, 0)
            hop_usage_sum["batches"].extend(hop_usage.get("batches") or [])
            for row in hop_rows:
                cid = int(row["id"])
                g = int(row.get("jev_grade", round(float(row["jev_score"]) * 10)))
                if g > best_hop.get(cid, 0):
                    best_hop[cid] = g

        # Layer-3: per-passage verifier on high-risk relations (top-N by s1).
        verify_stats = {"ran": 0, "valid": 0, "invalid": 0, "uncertain": 0, "skipped": 0}
        hop_effective: dict[int, int] = dict(best_hop)
        rel = gate_info.get("relation")
        ent = gate_info.get("entity") or ""
        do_verify = (
            bool(getattr(self.planner, "passage_verify", False))
            and rel in VERIFY_RELATIONS
            and bool(ent)
        )
        if do_verify:
            # Rank by raw s2 (bridge) grade — those are the segments that can leak.
            ranked_cands = sorted(
                candidates,
                key=lambda c: (-best_hop.get(int(c["id"]), 0), int(c["id"])),
            )
            top_n = int(getattr(self.planner, "verify_top_n", 20) or 20)
            scale = float(getattr(self.planner, "uncertain_scale", 0.2) or 0.2)
            for c in ranked_cands[:top_n]:
                cid = int(c["id"])
                verify_stats["ran"] += 1
                label = await self.planner.verifier.judge(
                    entity=str(ent),
                    relation=str(rel),
                    cand_id=cid,
                    title=c.get("title") or "",
                    text=c.get("content") or c.get("passage") or "",
                )
                verify_stats[label] = verify_stats.get(label, 0) + 1
                raw = hop_effective.get(cid, 0)
                if label == "invalid":
                    hop_effective[cid] = 0
                elif label == "uncertain":
                    hop_effective[cid] = int(round(raw * scale))
                # valid: keep
            for c in ranked_cands[top_n:]:
                verify_stats["skipped"] += 1
                # Outside top-N: do not let unchecked s2 dominate — drop s2.
                hop_effective[int(c["id"])] = 0
            # If most positive-s2 passages are invalid, treat as soft conflict:
            # drop the whole bridge path this round (s1 untouched).
            positive = sum(1 for g in best_hop.values() if g > 0)
            if (
                positive > 0
                and verify_stats.get("invalid", 0) >= max(2, (verify_stats.get("valid", 0) + 1))
            ):
                hop_effective = {cid: 0 for cid in hop_effective}
                verify_stats["bridge_killed"] = True
            self.planner.trace.append(
                {
                    "stage": "passage_verify",
                    "hop": self.planner.current_hop,
                    "entity": ent,
                    "relation": rel,
                    "stats": dict(verify_stats),
                }
            )

        merged = []
        for row in orig_rows:
            cid = int(row["id"])
            orig_g = int(row.get("jev_grade", round(float(row["jev_score"]) * 10)))
            hop_g = hop_effective.get(cid, 0)
            if self.mode == "blend":
                final_g = snap_blend(hop_g, orig_g, self.alpha)
            else:
                final_g = max(orig_g, hop_g)
            new_row = dict(row)
            new_row["jev_grade"] = final_g
            new_row["jev_score"] = final_g / 10.0
            new_row["jev_grade_orig"] = orig_g
            new_row["jev_grade_hop"] = hop_g
            new_row["jev_grade_hop_raw"] = best_hop.get(cid, 0)
            merged.append(new_row)

        usage = {
            "requests": orig_usage.get("requests", 0) + hop_usage_sum["requests"],
            "judgments": orig_usage.get("judgments", 0) + hop_usage_sum["judgments"],
            "input_tokens": orig_usage.get("input_tokens", 0) + hop_usage_sum["input_tokens"],
            "output_tokens": orig_usage.get("output_tokens", 0)
            + hop_usage_sum["output_tokens"],
            "imputed_judgments": orig_usage.get("imputed_judgments", 0)
            + hop_usage_sum["imputed_judgments"],
            "batches": list(orig_usage.get("batches") or []) + hop_usage_sum["batches"],
            "scoring_scale": orig_usage.get("scoring_scale", "grade"),
            "dual_anchor": {
                "mode": self.mode,
                "alpha": self.alpha,
                "anchors": anchors,
                "applied": True,
                "hop": self.planner.current_hop,
                "gate": gate_info,
                "passage_verify": verify_stats if do_verify else {"ran": 0},
            },
        }
        return merged, usage
