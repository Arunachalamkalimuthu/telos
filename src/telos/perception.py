"""Perception module: YOLOv8-nano object detection → WorldState construction."""

from __future__ import annotations

from typing import Any

from telos.world import Entity, Relation, WorldState

_DEFAULT_MODEL = None


def _get_model():
    """Lazy-load the YOLOv8-nano model on first call."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        from ultralytics import YOLO

        _DEFAULT_MODEL = YOLO("yolov8n.pt")
    return _DEFAULT_MODEL


def detect_objects(
    image_path: str,
    model: Any = None,
    confidence_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Run YOLOv8-nano on *image_path* and return a list of detections.

    Each detection is a dict with keys:
        label (str), confidence (float), bbox (tuple of four floats x1,y1,x2,y2).
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
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                }
            )
    return detections


def _bbox_distance(a: tuple, b: tuple) -> float:
    """Minimum axis-aligned distance between two bounding boxes.

    Returns 0.0 if the boxes overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    dx = max(0.0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0.0, max(ay1 - by2, by1 - ay2))
    return (dx ** 2 + dy ** 2) ** 0.5


def _overlap_area(a: tuple, b: tuple) -> float:
    """Area of intersection between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

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
) -> list[Relation]:
    """Derive spatial relations from a list of detections.

    Relations produced:
        ON(A, B)       — A's bottom edge is near B's top edge and A is horizontally within B.
        NEAR(A, B)     — bbox distance < near_threshold (each unordered pair appears once).
        CONTAINS(A, B) — B's bbox is mostly inside A's bbox (overlap/B_area >= containment_ratio).
    """
    n = len(detections)
    relations: list[Relation] = []

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

            # ON: A sits on top of B
            # A's bottom edge (ay2) is near B's top edge (by1)
            # and A is horizontally within B
            if (
                abs(ay2 - by1) <= on_tolerance
                and ax1 >= bx1 - on_tolerance
                and ax2 <= bx2 + on_tolerance
            ):
                relations.append(Relation("ON", a_id, b_id))

            # CONTAINS: A contains B
            b_area = _bbox_area(b_bbox)
            if b_area > 0:
                overlap = _overlap_area(a_bbox, b_bbox)
                if overlap / b_area >= containment_ratio:
                    relations.append(Relation("CONTAINS", a_id, b_id))

        # NEAR: only add once per unordered pair (i < j)
        for j in range(i + 1, n):
            b_id = f"{detections[j]['label']}_{j}"
            b_bbox = detections[j]["bbox"]
            dist = _bbox_distance(a_bbox, b_bbox)
            if dist < near_threshold:
                relations.append(Relation("NEAR", a_id, b_id))

    return relations


def build_world(image_path: str, model: Any = None) -> WorldState:
    """Detect objects, extract spatial relations, and return a WorldState."""
    detections = detect_objects(image_path, model=model)
    relations = extract_relations(detections)

    entities: dict[str, Entity] = {}
    for i, det in enumerate(detections):
        eid = f"{det['label']}_{i}"
        entities[eid] = Entity(
            id=eid,
            type=det["label"],
            properties={"confidence": det["confidence"], "bbox": det["bbox"]},
        )

    return WorldState(entities=entities, relations=tuple(relations))
