"""Perception: image/video → WorldState via YOLOv8-nano.

Three capabilities:
1. **Object detection** with physics property inference from a knowledge base
2. **Depth-aware spatial relations** using heuristic monocular depth from bbox geometry
3. **Video processing** with IoU-based object tracking across frames
"""

from __future__ import annotations

from typing import Any

from .world import Entity, Relation, WorldState


# ---------------------------------------------------------------------------
# YOLO model management
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = None


def _get_model():
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        from ultralytics import YOLO
        _DEFAULT_MODEL = YOLO("yolov8n.pt")
    return _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Physics property knowledge base
# ---------------------------------------------------------------------------

# Maps COCO class labels to telos physics properties.
# Properties match what the physics primitives in physics.py expect.
PROPERTY_KB: dict[str, dict[str, Any]] = {
    # Containers
    "cup": {"mass": 0.25, "fragile": True, "material": "ceramic", "impact_threshold": 2.0},
    "bowl": {"mass": 0.3, "fragile": True, "material": "ceramic", "impact_threshold": 2.0},
    "wine glass": {"mass": 0.15, "fragile": True, "material": "glass", "impact_threshold": 1.0},
    "bottle": {"mass": 0.4, "fragile": True, "material": "glass", "impact_threshold": 3.0},
    "vase": {"mass": 0.5, "fragile": True, "material": "ceramic", "impact_threshold": 1.5},
    # Electronics
    "laptop": {"mass": 1.8, "electronic": True, "fragile": True, "impact_threshold": 3.0},
    "cell phone": {"mass": 0.2, "electronic": True, "fragile": True, "impact_threshold": 2.0},
    "remote": {"mass": 0.15, "electronic": True},
    "keyboard": {"mass": 0.5, "electronic": True},
    "mouse": {"mass": 0.1, "electronic": True},
    "tv": {"mass": 8.0, "electronic": True, "fragile": True, "impact_threshold": 2.0},
    # Furniture / surfaces
    "dining table": {"mass": 30.0, "material": "wood"},
    "chair": {"mass": 5.0, "material": "wood"},
    "bench": {"mass": 15.0, "material": "wood"},
    "bed": {"mass": 40.0, "material": "fabric"},
    "couch": {"mass": 35.0, "material": "fabric"},
    # People / animals
    "person": {"mass": 70.0},
    "cat": {"mass": 4.0},
    "dog": {"mass": 15.0},
    "bird": {"mass": 0.5},
    # Vehicles
    "car": {"mass": 1500.0},
    "bicycle": {"mass": 10.0},
    "motorcycle": {"mass": 200.0},
    "bus": {"mass": 12000.0},
    "truck": {"mass": 8000.0},
    # Food / objects
    "apple": {"mass": 0.2},
    "orange": {"mass": 0.2},
    "banana": {"mass": 0.12},
    "sandwich": {"mass": 0.3},
    "pizza": {"mass": 0.8},
    "cake": {"mass": 1.0, "fragile": True, "impact_threshold": 0.5},
    "knife": {"mass": 0.1, "material": "metal"},
    "fork": {"mass": 0.05, "material": "metal"},
    "spoon": {"mass": 0.04, "material": "metal"},
    "scissors": {"mass": 0.08, "material": "metal"},
    "book": {"mass": 0.5},
    "clock": {"mass": 0.3, "fragile": True, "impact_threshold": 2.0},
    "umbrella": {"mass": 0.4},
    "backpack": {"mass": 0.8},
    "suitcase": {"mass": 3.0},
}


def get_physics_properties(label: str) -> dict[str, Any]:
    """Look up physics properties for a COCO class label.

    Returns known properties from PROPERTY_KB, or an empty dict for
    unrecognized labels.
    """
    return dict(PROPERTY_KB.get(label, {}))


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def detect_objects(
    image_path: str,
    model: Any = None,
    confidence_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Run YOLOv8-nano on an image and return detections.

    Each detection is a dict with keys:
        label, confidence, bbox (x1, y1, x2, y2).
    """
    if model is None:
        model = _get_model()

    results = model(image_path, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        for i in range(len(confs)):
            conf = float(confs[i])
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = xyxy[i]
            label = model.names[int(classes[i])]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })
    return detections


# ---------------------------------------------------------------------------
# Heuristic depth estimation
# ---------------------------------------------------------------------------

def estimate_depth(
    detections: list[dict[str, Any]],
    image_height: float = 480.0,
) -> list[dict[str, Any]]:
    """Add heuristic depth estimates to detections based on bbox geometry.

    Uses two cues from monocular perspective:
    - **Vertical position:** objects lower in the image are closer (ground plane assumption)
    - **Relative size:** larger bounding boxes suggest closer objects

    Adds 'depth' (float, 0.0=closest, 1.0=farthest) to each detection dict.
    Returns a new list (does not mutate input).
    """
    if not detections:
        return []

    enriched = [dict(d) for d in detections]

    # Compute raw depth signals.
    max_area = 0.0
    for d in enriched:
        x1, y1, x2, y2 = d["bbox"]
        area = (x2 - x1) * (y2 - y1)
        d["_area"] = area
        d["_bottom_y"] = y2
        if area > max_area:
            max_area = area

    if max_area == 0:
        for d in enriched:
            d["depth"] = 0.5
            d.pop("_area", None)
            d.pop("_bottom_y", None)
        return enriched

    for d in enriched:
        # Vertical: bottom_y closer to image_height → closer → lower depth value.
        vert_depth = 1.0 - (d["_bottom_y"] / image_height)
        # Size: larger area → closer → lower depth value.
        size_depth = 1.0 - (d["_area"] / max_area)
        # Combine: weighted average (vertical position is more reliable).
        d["depth"] = 0.6 * vert_depth + 0.4 * size_depth
        d.pop("_area", None)
        d.pop("_bottom_y", None)

    return enriched


# ---------------------------------------------------------------------------
# Spatial relation extraction (depth-aware)
# ---------------------------------------------------------------------------

def _bbox_distance(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0.0, max(ay1 - by2, by1 - ay2))
    return (dx ** 2 + dy ** 2) ** 0.5


def _overlap_area(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _bbox_area(bbox: tuple) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def extract_relations(
    detections: list[dict[str, Any]],
    near_threshold: float = 50.0,
    on_tolerance: float = 30.0,
    containment_ratio: float = 0.7,
    depth_threshold: float = 0.15,
) -> list[Relation]:
    """Derive spatial relations from detections with optional depth awareness.

    Relations:
    - ON(A, B): A sits on top of B (A's bottom near B's top, horizontally within B)
    - NEAR(A, B): bbox distance < threshold (unordered, once per pair)
    - CONTAINS(A, B): B mostly inside A
    - IN_FRONT_OF(A, B): A is significantly closer than B (requires depth estimates)
    - BEHIND(A, B): A is significantly farther than B (requires depth estimates)
    """
    n = len(detections)
    relations: list[Relation] = []
    has_depth = n > 0 and "depth" in detections[0]

    for i in range(n):
        a_id = f"{detections[i]['label']}_{i}"
        a_bbox = detections[i]["bbox"]
        ax1, ay1, ax2, ay2 = a_bbox

        for j in range(n):
            if i == j:
                continue
            b_id = f"{detections[j]['label']}_{j}"
            b_bbox = detections[j]["bbox"]
            bx1, by1, bx2, by2 = b_bbox

            # ON: A's bottom near B's top, A horizontally within B.
            if (
                abs(ay2 - by1) <= on_tolerance
                and ax1 >= bx1 - on_tolerance
                and ax2 <= bx2 + on_tolerance
            ):
                relations.append(Relation("ON", a_id, b_id))

            # CONTAINS: B mostly inside A.
            b_area = _bbox_area(b_bbox)
            if b_area > 0:
                overlap = _overlap_area(a_bbox, b_bbox)
                if overlap / b_area >= containment_ratio:
                    relations.append(Relation("CONTAINS", a_id, b_id))

        # NEAR + depth relations: once per unordered pair.
        for j in range(i + 1, n):
            b_id = f"{detections[j]['label']}_{j}"
            b_bbox = detections[j]["bbox"]
            dist = _bbox_distance(a_bbox, b_bbox)
            if dist < near_threshold:
                relations.append(Relation("NEAR", a_id, b_id))

            # Depth-based relations.
            if has_depth:
                depth_a = detections[i].get("depth", 0.5)
                depth_b = detections[j].get("depth", 0.5)
                diff = depth_a - depth_b
                if diff < -depth_threshold:
                    relations.append(Relation("IN_FRONT_OF", a_id, b_id))
                elif diff > depth_threshold:
                    relations.append(Relation("BEHIND", a_id, b_id))

    return relations


# ---------------------------------------------------------------------------
# World building (single image)
# ---------------------------------------------------------------------------

def build_world(
    image_path: str,
    model: Any = None,
    use_depth: bool = True,
    use_properties: bool = True,
    image_height: float | None = None,
) -> WorldState:
    """End-to-end: image → detect → depth → properties → relations → WorldState.

    Args:
        image_path: Path to the image file.
        model: Optional pre-loaded YOLO model.
        use_depth: If True, estimate depth and include depth-aware relations.
        use_properties: If True, enrich entities with physics properties from KB.
        image_height: Image height for depth estimation. Auto-detected if None.
    """
    detections = detect_objects(image_path, model=model)

    if use_depth:
        if image_height is None:
            # Estimate from max bbox y-coordinate.
            if detections:
                image_height = max(d["bbox"][3] for d in detections) * 1.1
            else:
                image_height = 480.0
        detections = estimate_depth(detections, image_height=image_height)

    relations = extract_relations(detections)

    entities: dict[str, Entity] = {}
    for i, det in enumerate(detections):
        eid = f"{det['label']}_{i}"
        props: dict[str, Any] = {
            "confidence": det["confidence"],
            "bbox": det["bbox"],
        }
        if "depth" in det:
            props["depth"] = det["depth"]
        if use_properties:
            physics = get_physics_properties(det["label"])
            props.update(physics)

        entities[eid] = Entity(id=eid, type=det["label"], properties=props)

    return WorldState(entities=entities, relations=tuple(relations))


# ---------------------------------------------------------------------------
# Video processing with IoU tracking
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two bounding boxes."""
    overlap = _overlap_area(a, b)
    if overlap == 0:
        return 0.0
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - overlap
    if union == 0:
        return 0.0
    return overlap / union


def track_objects(
    prev_detections: list[dict[str, Any]],
    curr_detections: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[int, int | None]:
    """Match current detections to previous detections via IoU.

    Returns a mapping from current detection index to previous detection index
    (or None if the object is new).
    """
    matches: dict[int, int | None] = {}
    used_prev: set[int] = set()

    # Compute IoU matrix.
    scores: list[tuple[float, int, int]] = []
    for ci, curr in enumerate(curr_detections):
        for pi, prev in enumerate(prev_detections):
            if curr["label"] != prev["label"]:
                continue
            score = _iou(curr["bbox"], prev["bbox"])
            if score >= iou_threshold:
                scores.append((score, ci, pi))

    # Greedy matching: highest IoU first.
    scores.sort(reverse=True)
    used_curr: set[int] = set()
    for score, ci, pi in scores:
        if ci in used_curr or pi in used_prev:
            continue
        matches[ci] = pi
        used_curr.add(ci)
        used_prev.add(pi)

    # Unmatched current detections are new objects.
    for ci in range(len(curr_detections)):
        if ci not in matches:
            matches[ci] = None

    return matches


def process_video(
    frame_paths: list[str],
    model: Any = None,
    use_depth: bool = True,
    use_properties: bool = True,
    iou_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Process a sequence of frames and return tracked world states.

    Returns a list of dicts, one per frame:
        {
            "frame": int,
            "world": WorldState,
            "tracks": {curr_entity_id: prev_entity_id | "new"},
            "events": ["appeared: ...", "disappeared: ...", "moved: ..."],
        }
    """
    if not frame_paths:
        return []

    results: list[dict[str, Any]] = []
    prev_detections: list[dict[str, Any]] = []
    prev_entity_ids: list[str] = []

    for frame_idx, path in enumerate(frame_paths):
        detections = detect_objects(path, model=model)

        if use_depth:
            if detections:
                img_h = max(d["bbox"][3] for d in detections) * 1.1
            else:
                img_h = 480.0
            detections = estimate_depth(detections, image_height=img_h)

        relations = extract_relations(detections)

        # Build entities with properties.
        entities: dict[str, Entity] = {}
        curr_entity_ids: list[str] = []
        for i, det in enumerate(detections):
            eid = f"{det['label']}_{i}"
            props: dict[str, Any] = {
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
            if "depth" in det:
                props["depth"] = det["depth"]
            if use_properties:
                props.update(get_physics_properties(det["label"]))
            entities[eid] = Entity(id=eid, type=det["label"], properties=props)
            curr_entity_ids.append(eid)

        world = WorldState(
            entities=entities,
            relations=tuple(relations),
            time=float(frame_idx),
        )

        # Track objects across frames.
        tracks: dict[str, str | None] = {}
        events: list[str] = []

        if frame_idx > 0 and prev_detections:
            matching = track_objects(prev_detections, detections, iou_threshold)
            matched_prev: set[int] = set()

            for ci, pi in matching.items():
                curr_eid = curr_entity_ids[ci] if ci < len(curr_entity_ids) else None
                if curr_eid is None:
                    continue
                if pi is not None:
                    prev_eid = prev_entity_ids[pi] if pi < len(prev_entity_ids) else None
                    tracks[curr_eid] = prev_eid
                    matched_prev.add(pi)

                    # Detect movement.
                    if prev_eid and pi < len(prev_detections):
                        prev_bbox = prev_detections[pi]["bbox"]
                        curr_bbox = detections[ci]["bbox"]
                        dx = abs((curr_bbox[0] + curr_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2)
                        dy = abs((curr_bbox[1] + curr_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2)
                        if dx > 20 or dy > 20:
                            events.append(f"moved: {curr_eid}")
                else:
                    tracks[curr_eid] = None
                    events.append(f"appeared: {curr_eid}")

            # Disappeared objects.
            for pi in range(len(prev_detections)):
                if pi not in matched_prev:
                    prev_eid = prev_entity_ids[pi] if pi < len(prev_entity_ids) else f"unknown_{pi}"
                    events.append(f"disappeared: {prev_eid}")

        results.append({
            "frame": frame_idx,
            "world": world,
            "tracks": tracks,
            "events": events,
        })

        prev_detections = detections
        prev_entity_ids = curr_entity_ids

    return results
