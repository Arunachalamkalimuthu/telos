"""Natural-language understanding: text → WorldState / structured queries.

Three capabilities beyond basic parsing:
1. **Negation & quantifier handling** — "not fragile", "no liquid", "all cups"
2. **Physics property mapping** — adjectives like "ceramic" → {fragile: True, material: "ceramic"}
3. **Executable queries** — parse_query output wires directly into CausalGraph.do() / counterfactual()
"""

from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.tokens import Doc

from .causal_graph import CausalGraph
from .world import Entity, Relation, WorldState


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
# Physics property mapping (adjective → properties)
# ---------------------------------------------------------------------------

# Maps adjectives and material nouns to physics properties that the telos
# physics primitives understand. Each key is a lowercased token; the value
# is a dict of properties to merge into the entity.
ADJECTIVE_PROPERTY_MAP: dict[str, dict[str, Any]] = {
    # Materials → fragile, material
    "ceramic": {"fragile": True, "material": "ceramic", "impact_threshold": 2.0},
    "glass": {"fragile": True, "material": "glass", "impact_threshold": 1.0},
    "wooden": {"material": "wood"},
    "wood": {"material": "wood"},
    "metal": {"material": "metal"},
    "metallic": {"material": "metal"},
    "plastic": {"material": "plastic"},
    "paper": {"material": "paper"},
    "fabric": {"material": "fabric", "absorbent": True},
    "cloth": {"material": "fabric", "absorbent": True},
    "stone": {"material": "stone"},
    "rubber": {"material": "rubber"},
    # Physical properties
    "heavy": {"mass_hint": "heavy"},
    "light": {"mass_hint": "light"},
    "fragile": {"fragile": True},
    "sturdy": {"fragile": False},
    "breakable": {"fragile": True},
    "unbreakable": {"fragile": False},
    "absorbent": {"absorbent": True},
    # State
    "hot": {"temperature": "hot"},
    "cold": {"temperature": "cold"},
    "wet": {"wet": True},
    "dry": {"wet": False},
    "open": {"sealed": False},
    "closed": {"sealed": True},
    "sealed": {"sealed": True},
    "unsealed": {"sealed": False},
    "inverted": {"orientation": "inverted"},
    "upright": {"orientation": "upright"},
    "upside-down": {"orientation": "inverted"},
    "empty": {"contains": None},
    "full": {"full": True},
    # Electronic
    "electronic": {"electronic": True},
    "electric": {"electronic": True},
    "digital": {"electronic": True},
    # Liquid properties
    "conductive": {"conductive": True},
    "liquid": {"type_hint": "liquid"},
}

# Common nouns that imply physics properties on adjacent entities.
NOUN_PROPERTY_MAP: dict[str, dict[str, Any]] = {
    "cup": {"mass": 0.25},
    "laptop": {"electronic": True, "mass": 1.8},
    "phone": {"electronic": True, "mass": 0.2},
    "table": {"mass": 30.0},
    "bottle": {"mass": 0.4},
    "glass": {"fragile": True, "mass": 0.15},
    "bowl": {"mass": 0.3},
    "vase": {"fragile": True, "mass": 0.5},
    "knife": {"mass": 0.1, "material": "metal"},
    "book": {"mass": 0.5},
    "person": {"mass": 70.0},
    "child": {"mass": 25.0},
    "car": {"mass": 1500.0},
    "water": {"conductive": True, "type_hint": "liquid"},
    "coffee": {"conductive": True, "type_hint": "liquid"},
    "tea": {"conductive": True, "type_hint": "liquid"},
    "juice": {"conductive": True, "type_hint": "liquid"},
    "milk": {"conductive": True, "type_hint": "liquid"},
    "oil": {"conductive": False, "type_hint": "liquid"},
}


def map_properties(entity_type: str, attributes: list[str]) -> dict[str, Any]:
    """Map an entity type and its adjective attributes to physics properties.

    Merges properties from NOUN_PROPERTY_MAP (by type) and
    ADJECTIVE_PROPERTY_MAP (by each attribute). Adjective properties
    override noun defaults.
    """
    props: dict[str, Any] = {}

    # Base properties from noun type.
    noun_props = NOUN_PROPERTY_MAP.get(entity_type, {})
    props.update(noun_props)

    # Adjective overrides.
    for attr in attributes:
        adj_props = ADJECTIVE_PROPERTY_MAP.get(attr, {})
        props.update(adj_props)

    return props


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

def _is_negated(token) -> bool:
    """Check if a token is negated via a 'neg' dependency child."""
    for child in token.children:
        if child.dep_ == "neg":
            return True
    # Also check if the token's head has negation and this is its subject/object.
    if token.head and token.head != token:
        for child in token.head.children:
            if child.dep_ == "neg":
                return True
    return False


def _detect_negation(doc: Doc) -> set[int]:
    """Return token indices that are in the scope of a negation.

    Detects patterns like:
    - "not fragile" (neg → amod)
    - "no liquid" (det neg → noun)
    - "isn't sealed" (neg → verb/adj)
    """
    negated_indices: set[int] = set()

    for token in doc:
        if token.dep_ == "neg":
            # The head of the negation and its subtree are negated.
            head = token.head
            negated_indices.add(head.i)
            # Also mark adjective/noun children of the negated head.
            for child in head.children:
                if child.dep_ in ("acomp", "attr", "amod", "dobj"):
                    negated_indices.add(child.i)

        # "no" as determiner negates its head noun.
        if token.text.lower() == "no" and token.dep_ == "det":
            negated_indices.add(token.head.i)

    return negated_indices


# ---------------------------------------------------------------------------
# Quantifier detection
# ---------------------------------------------------------------------------

def detect_quantifiers(doc: Doc) -> dict[int, str]:
    """Return a mapping from noun token indices to their quantifier.

    Detects: all, every, each, some, any, no, many, few, several.
    """
    quantifiers: dict[int, str] = {}
    _quant_words = {"all", "every", "each", "some", "any", "no", "many", "few", "several"}

    for token in doc:
        if token.text.lower() in _quant_words and token.dep_ in ("det", "predet"):
            quantifiers[token.head.i] = token.text.lower()

    return quantifiers


# ---------------------------------------------------------------------------
# Scene parsing helpers
# ---------------------------------------------------------------------------

def _noun_chunk_for_token(doc: Doc, token) -> Any | None:
    for nc in doc.noun_chunks:
        if nc.start <= token.i < nc.end:
            return nc
    return None


def _adjective_modifiers(chunk) -> list[str]:
    return [tok.text.lower() for tok in chunk if tok.dep_ == "amod" and tok.pos_ == "ADJ"]


def _find_subject_for_prep(doc: Doc, prep_token):
    head = prep_token.head
    if head.pos_ == "NOUN":
        return head
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


# ---------------------------------------------------------------------------
# Scene parsing
# ---------------------------------------------------------------------------

def parse_scene(text: str, enrich_properties: bool = True) -> WorldState:
    """Parse a natural-language scene description into a WorldState.

    Uses spaCy dependency parsing:
    - Noun chunks → entities (root noun lemma = type, ADJ children = attributes)
    - ``prep`` + ``pobj`` pattern with spatial lemmas → relations
    - Negation detection marks negated properties
    - If ``enrich_properties`` is True, adjectives are mapped to physics properties

    Examples:
        >>> parse_scene("A ceramic cup is on a wooden table")
        # cup has fragile=True, material="ceramic"; table has material="wood"

        >>> parse_scene("The cup is not sealed")
        # cup has sealed=False via negation
    """
    nlp = load_model()
    doc = nlp(text)

    negated = _detect_negation(doc)
    quantifiers = detect_quantifiers(doc)

    # --- 1. Collect entities from noun chunks ---
    entities: dict[str, Entity] = {}
    type_counts: dict[str, int] = {}
    token_to_entity: dict[int, str] = {}

    for chunk in doc.noun_chunks:
        root = chunk.root
        etype = root.lemma_.lower()
        attrs = _adjective_modifiers(chunk)

        # Handle negated adjectives: if the adj token is in negated set,
        # prefix with "not_" so downstream can interpret.
        resolved_attrs: list[str] = []
        for tok in chunk:
            if tok.dep_ == "amod" and tok.pos_ == "ADJ":
                if tok.i in negated:
                    resolved_attrs.append(f"not_{tok.text.lower()}")
                else:
                    resolved_attrs.append(tok.text.lower())

        # Deduplicate ids.
        type_counts[etype] = type_counts.get(etype, 0) + 1
        count = type_counts[etype]
        eid = etype if count == 1 else f"{etype}_{count}"

        props: dict[str, Any] = {}
        if resolved_attrs:
            props["attributes"] = resolved_attrs

        # Map adjectives to physics properties.
        if enrich_properties:
            physics = map_properties(etype, [a for a in resolved_attrs if not a.startswith("not_")])

            # Handle negated adjectives: if "not_fragile", set fragile=False.
            for attr in resolved_attrs:
                if attr.startswith("not_"):
                    base = attr[4:]
                    adj_props = ADJECTIVE_PROPERTY_MAP.get(base, {})
                    for key, val in adj_props.items():
                        if isinstance(val, bool):
                            physics[key] = not val

            props.update(physics)

        # Store quantifier if present.
        if root.i in quantifiers:
            props["quantifier"] = quantifiers[root.i]

        entities[eid] = Entity(id=eid, type=etype, properties=props)
        for tok in chunk:
            token_to_entity[tok.i] = eid

    # --- Handle predicate adjectives / passive verbs ---
    # "the cup is sealed" → acomp (ADJ) or passive VERB with nsubjpass
    if enrich_properties:
        for token in doc:
            adj = token.text.lower()
            is_neg = token.i in negated
            subj = None

            if token.dep_ == "acomp" and token.pos_ == "ADJ":
                # Copula pattern: "X is ADJ"
                verb = token.head
                for child in verb.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = child
                        break
            elif token.pos_ == "VERB" and adj in ADJECTIVE_PROPERTY_MAP:
                # Passive pattern: "X is sealed" (sealed parsed as VERB)
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = child
                        break

            if subj is None:
                continue
            eid = token_to_entity.get(subj.i)
            if eid is None:
                continue

            adj_props = ADJECTIVE_PROPERTY_MAP.get(adj, {})
            if adj_props and eid in entities:
                old_props = dict(entities[eid].properties)
                for key, val in adj_props.items():
                    if is_neg and isinstance(val, bool):
                        old_props[key] = not val
                    else:
                        old_props[key] = val
                entities[eid] = Entity(
                    id=entities[eid].id,
                    type=entities[eid].type,
                    properties=old_props,
                )

    # --- 2. Extract relations via prep→pobj arcs ---
    relations: list[Relation] = []
    for token in doc:
        if token.dep_ == "prep" and token.lemma_.lower() in _SPATIAL_MAP:
            rel_name = _SPATIAL_MAP[token.lemma_.lower()]
            obj_token = None
            for child in token.children:
                if child.dep_ == "pobj":
                    obj_token = child
                    break
            if obj_token is None:
                continue

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
    r"suppose",
    r"imagine\s+(?:that\s+)?",
    r"assuming",
]

_PREDICTION_PATTERNS = [
    r"^will\s+",
    r"is\s+it\s+going\s+to",
    r"^is\s+the\s+",
    r"^are\s+the\s+",
    r"^does\s+the\s+",
    r"^can\s+the\s+",
    r"^could\s+the\s+",
]

_AUX_VERBS = {"be", "do", "have", "happen", "would", "will", "can", "get", "could", "should"}
_QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which"}

# Maps natural language action verbs/adjectives to causal variable suffixes.
_ACTION_TO_VARIABLE: dict[str, str] = {
    "fall": "falls",
    "break": "breaks",
    "spill": "contents_escape",
    "escape": "contents_escape",
    "damage": "damaged",
    "seal": "sealed",
    "open": "sealed",
    "invert": "inverted",
}

# Maps natural language state adjectives to variable suffixes + values.
_STATE_TO_INTERVENTION: dict[str, tuple[str, Any]] = {
    "sealed": ("sealed", True),
    "unsealed": ("sealed", False),
    "open": ("sealed", False),
    "closed": ("sealed", True),
    "inverted": ("inverted", True),
    "upright": ("inverted", False),
    "broken": ("breaks", True),
    "damaged": ("damaged", True),
    "safe": ("damaged", False),
    "fallen": ("falls", True),
    "spilled": ("contents_escape", True),
}


def parse_query(text: str) -> dict[str, Any]:
    """Parse a natural-language question into a structured query.

    Returns a dict with:
    - ``type``: "counterfactual" or "prediction"
    - ``subject``: the main entity mentioned (str or None)
    - ``action``: the verb/event mentioned (str or None)
    - ``intervention``: for counterfactuals, a dict mapping variable names to values
      that can be passed to ``CausalGraph.counterfactual()``. Example:
      ``{"cup.contents_escape": False}``
    - ``target``: for predictions, the variable to check. Example: ``"laptop.damaged"``
    """
    nlp = load_model()
    doc = nlp(text)
    lower = text.lower()

    # --- Determine type ---
    qtype = "prediction"
    for pat in _COUNTERFACTUAL_PATTERNS:
        if re.search(pat, lower):
            qtype = "counterfactual"
            break
    else:
        for pat in _PREDICTION_PATTERNS:
            if re.search(pat, lower):
                qtype = "prediction"
                break

    # --- Check for negation in the query ---
    negated = _detect_negation(doc)

    # --- Extract subject: first non-question-word noun chunk ---
    subject: str | None = None
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() not in _QUESTION_WORDS:
            subject = chunk.root.lemma_.lower()
            break

    # --- Extract action: first meaningful verb ---
    action: str | None = None
    action_token = None
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() not in _AUX_VERBS:
            action = token.lemma_.lower()
            action_token = token
            break

    # --- Also check for state adjectives (e.g., "the cup were sealed") ---
    state_adj: str | None = None
    for token in doc:
        if token.pos_ == "ADJ" and token.lemma_.lower() in _STATE_TO_INTERVENTION:
            state_adj = token.lemma_.lower()
            break

    # --- Build intervention / target ---
    intervention: dict[str, Any] = {}
    target: str | None = None

    if qtype == "counterfactual" and subject:
        if state_adj and state_adj in _STATE_TO_INTERVENTION:
            var_suffix, value = _STATE_TO_INTERVENTION[state_adj]
            # Check for negation on the adjective.
            if state_adj and any(
                tok.lemma_.lower() == state_adj and tok.i in negated
                for tok in doc
            ):
                if isinstance(value, bool):
                    value = not value
            intervention[f"{subject}.{var_suffix}"] = value
        elif action and action in _ACTION_TO_VARIABLE:
            var_suffix = _ACTION_TO_VARIABLE[action]
            is_neg = action_token is not None and action_token.i in negated
            intervention[f"{subject}.{var_suffix}"] = not is_neg
        elif action:
            intervention[f"{subject}.{action}"] = True

    if qtype == "prediction" and subject:
        if action and action in _ACTION_TO_VARIABLE:
            var_suffix = _ACTION_TO_VARIABLE[action]
            target = f"{subject}.{var_suffix}"
        elif state_adj and state_adj in _STATE_TO_INTERVENTION:
            var_suffix, _ = _STATE_TO_INTERVENTION[state_adj]
            target = f"{subject}.{var_suffix}"
        elif action:
            target = f"{subject}.{action}"

    result: dict[str, Any] = {
        "type": qtype,
        "subject": subject,
        "action": action,
    }
    if intervention:
        result["intervention"] = intervention
    if target:
        result["target"] = target

    return result


# ---------------------------------------------------------------------------
# Executable query runner
# ---------------------------------------------------------------------------

def execute_query(
    query: dict[str, Any],
    graph: CausalGraph,
) -> dict[str, Any]:
    """Execute a parsed query against a CausalGraph.

    For counterfactual queries, applies interventions via ``graph.counterfactual()``.
    For prediction queries, propagates the graph and returns the target variable value.

    Returns a dict with:
    - ``type``: the query type
    - ``result``: the propagated state (dict) or target value
    - ``intervention``: the interventions applied (for counterfactuals)
    - ``target``: the target variable (for predictions)
    """
    qtype = query.get("type", "prediction")

    if qtype == "counterfactual":
        intervention = query.get("intervention", {})
        # Filter to only variables that exist in the graph.
        valid_interventions = {
            k: v for k, v in intervention.items() if k in graph.variables()
        }
        if valid_interventions:
            state = graph.counterfactual(valid_interventions)
        else:
            state = graph.propagate()
        return {
            "type": "counterfactual",
            "intervention": valid_interventions,
            "result": state,
        }
    else:
        state = graph.propagate()
        target = query.get("target")
        if target and target in state:
            return {
                "type": "prediction",
                "target": target,
                "result": state[target],
                "full_state": state,
            }
        return {
            "type": "prediction",
            "target": target,
            "result": state,
        }
