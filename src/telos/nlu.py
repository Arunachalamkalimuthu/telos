"""Natural-language understanding: text → WorldState / structured queries."""

from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.tokens import Doc

from telos.world import Entity, Relation, WorldState

# ---------------------------------------------------------------------------
# spaCy model cache
# ---------------------------------------------------------------------------

_nlp: spacy.language.Language | None = None


def load_model() -> spacy.language.Language:
    """Load spaCy ``en_core_web_sm``, cached after first call."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Spatial-relation mapping
# ---------------------------------------------------------------------------

_SPATIAL_MAP: dict[str, str] = {
    "on": "ON",
    "above": "ABOVE",
    "below": "BELOW",
    "near": "NEAR",
    "beside": "NEAR",
    "next": "NEAR",
    "inside": "CONTAINS",
    "in": "CONTAINS",
    "under": "BELOW",
    "over": "ABOVE",
}

# ---------------------------------------------------------------------------
# Scene parsing
# ---------------------------------------------------------------------------


def _noun_chunk_for_token(doc: Doc, token) -> Any | None:
    """Return the noun-chunk that contains *token*, or ``None``."""
    for nc in doc.noun_chunks:
        if nc.start <= token.i < nc.end:
            return nc
    return None


def _adjective_modifiers(chunk) -> list[str]:
    """Return lowercased ADJ modifiers attached to the root of a noun chunk."""
    return [tok.text.lower() for tok in chunk if tok.dep_ == "amod" and tok.pos_ == "ADJ"]


def _find_subject_for_prep(doc: Doc, prep_token):
    """Walk up from a prep token to find the subject entity token.

    Strategy:
    - If the prep's head is a NOUN, return that noun directly (handles
      chained preps like "a table near a laptop").
    - Otherwise walk ancestors looking for a token that has an nsubj /
      nsubjpass child, and return that child.
    """
    head = prep_token.head
    if head.pos_ == "NOUN":
        return head
    # Walk the head and its ancestors
    visited = set()
    current = head
    while current is not None and current.i not in visited:
        visited.add(current.i)
        for child in current.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
        if current.head.i == current.i:
            break
        current = current.head
    return None


def parse_scene(text: str) -> WorldState:
    """Parse a natural-language scene description into a :class:`WorldState`.

    Uses spaCy dependency parsing:
    * Noun chunks → entities (root noun lemma = type, ADJ children = attributes).
    * ``prep`` + ``pobj`` dependency pattern with spatial lemmas → relations.
    """
    nlp = load_model()
    doc = nlp(text)

    # --- 1. Collect entities from noun chunks ---------------------------------
    entities: dict[str, Entity] = {}
    # Track how many times each type has been seen (for dedup)
    type_counts: dict[str, int] = {}
    # Map token index → entity id (for relation building)
    token_to_entity: dict[int, str] = {}

    for chunk in doc.noun_chunks:
        root = chunk.root
        etype = root.lemma_.lower()
        attrs = _adjective_modifiers(chunk)

        # Deduplicate ids
        type_counts[etype] = type_counts.get(etype, 0) + 1
        count = type_counts[etype]
        eid = etype if count == 1 else f"{etype}_{count}"

        props: dict[str, Any] = {}
        if attrs:
            props["attributes"] = attrs

        entities[eid] = Entity(id=eid, type=etype, properties=props)
        # Map every token in the chunk to this entity id
        for tok in chunk:
            token_to_entity[tok.i] = eid

    # --- 2. Extract relations via prep→pobj arcs ----------------------------
    relations: list[Relation] = []
    for token in doc:
        if token.dep_ == "prep" and token.lemma_.lower() in _SPATIAL_MAP:
            rel_name = _SPATIAL_MAP[token.lemma_.lower()]
            # Object: the pobj child of this prep
            obj_token = None
            for child in token.children:
                if child.dep_ == "pobj":
                    obj_token = child
                    break
            if obj_token is None:
                continue

            # Subject: walk up from prep
            subj_token = _find_subject_for_prep(doc, token)
            if subj_token is None:
                continue

            src_id = token_to_entity.get(subj_token.i)
            dst_id = token_to_entity.get(obj_token.i)
            if src_id and dst_id:
                relations.append(Relation(rel_name, src_id, dst_id))

    return WorldState(entities=entities, relations=tuple(relations))


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

_COUNTERFACTUAL_PATTERNS = [
    r"what\s+happens?\s+if",
    r"what\s+if",
    r"what\s+would\s+happen",
]

_PREDICTION_PATTERNS = [
    r"^will\s+",
    r"is\s+it\s+going\s+to",
    r"^is\s+the\s+",
    r"^are\s+the\s+",
    r"^does\s+the\s+",
]

_AUX_VERBS = {"be", "do", "have", "happen", "would", "will", "can", "get"}

_QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which"}


def parse_query(text: str) -> dict[str, Any]:
    """Parse a natural-language question into a structured dict.

    Returns ``{"type": "counterfactual"|"prediction", "subject": str|None, "action": str|None}``.
    """
    nlp = load_model()
    doc = nlp(text)
    lower = text.lower()

    # --- Determine type -------------------------------------------------------
    qtype = "prediction"  # default
    for pat in _COUNTERFACTUAL_PATTERNS:
        if re.search(pat, lower):
            qtype = "counterfactual"
            break
    else:
        for pat in _PREDICTION_PATTERNS:
            if re.search(pat, lower):
                qtype = "prediction"
                break

    # --- Extract subject: first non-question-word noun chunk ------------------
    subject: str | None = None
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() not in _QUESTION_WORDS:
            subject = chunk.root.lemma_.lower()
            break

    # --- Extract action: first meaningful verb --------------------------------
    action: str | None = None
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() not in _AUX_VERBS:
            action = token.lemma_.lower()
            break

    return {"type": qtype, "subject": subject, "action": action}
