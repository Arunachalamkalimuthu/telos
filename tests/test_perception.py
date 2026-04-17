"""Tests for the perception module."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from telos.perception import (
    build_world,
    detect_objects,
    estimate_depth,
    extract_relations,
    get_physics_properties,
    track_objects,
)
from telos.world import Relation


# ---------------------------------------------------------------------------
# Physics property KB
# ---------------------------------------------------------------------------

class TestPhysicsProperties(unittest.TestCase):

    def test_cup_has_physics_properties(self):
        props = get_physics_properties("cup")
        self.assertIn("mass", props)
        self.assertIn("fragile", props)
        self.assertTrue(props["fragile"])

    def test_laptop_is_electronic(self):
        props = get_physics_properties("laptop")
        self.assertTrue(props["electronic"])
        self.assertIn("mass", props)

    def test_unknown_label_returns_empty(self):
        props = get_physics_properties("frambulator")
        self.assertEqual(props, {})

    def test_returns_copy_not_reference(self):
        """Mutating the result should not affect the KB."""
        props = get_physics_properties("cup")
        props["mass"] = 999
        self.assertNotEqual(get_physics_properties("cup")["mass"], 999)


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

class TestDetectObjects(unittest.TestCase):

    def test_single_detection(self):
        mock_model = MagicMock()
        mock_model.names = {41: "cup"}
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [[100.0, 200.0, 300.0, 400.0]]
        )
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([41.0])
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detections = detect_objects("fake.jpg", model=mock_model)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "cup")
        self.assertAlmostEqual(detections[0]["confidence"], 0.95)

    def test_below_threshold_filtered(self):
        mock_model = MagicMock()
        mock_model.names = {41: "cup"}
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 20.0, 30.0, 40.0]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.1])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([41.0])
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detections = detect_objects("fake.jpg", model=mock_model, confidence_threshold=0.3)
        self.assertEqual(len(detections), 0)


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

class TestEstimateDepth(unittest.TestCase):

    def test_lower_objects_have_lower_depth(self):
        """Objects lower in image (closer) should have lower depth values."""
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 50, 200, 100)},   # high in image (far)
            {"label": "cup", "confidence": 0.9, "bbox": (100, 350, 200, 450)},  # low in image (close)
        ]
        result = estimate_depth(detections, image_height=480.0)
        self.assertLess(result[1]["depth"], result[0]["depth"])

    def test_larger_objects_have_lower_depth(self):
        """Larger bounding boxes (closer) should have lower depth values."""
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 200, 150, 250)},  # small (far)
            {"label": "cup", "confidence": 0.9, "bbox": (100, 200, 300, 400)},  # large (close)
        ]
        result = estimate_depth(detections, image_height=480.0)
        self.assertLess(result[1]["depth"], result[0]["depth"])

    def test_empty_detections(self):
        result = estimate_depth([])
        self.assertEqual(result, [])

    def test_does_not_mutate_input(self):
        detections = [{"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)}]
        estimate_depth(detections)
        self.assertNotIn("depth", detections[0])

    def test_depth_in_range(self):
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (10, 10, 100, 100)},
            {"label": "laptop", "confidence": 0.9, "bbox": (200, 300, 400, 450)},
        ]
        result = estimate_depth(detections, image_height=480.0)
        for d in result:
            self.assertGreaterEqual(d["depth"], 0.0)
            self.assertLessEqual(d["depth"], 1.0)


# ---------------------------------------------------------------------------
# Spatial relations (including depth-aware)
# ---------------------------------------------------------------------------

class TestExtractRelations(unittest.TestCase):

    def test_on_relation(self):
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (150, 80, 250, 150)},
            {"label": "table", "confidence": 0.9, "bbox": (50, 150, 400, 300)},
        ]
        relations = extract_relations(detections)
        on_rels = [r for r in relations if r.name == "ON"]
        self.assertTrue(any(r.src == "cup_0" and r.dst == "table_1" for r in on_rels))

    def test_near_relation(self):
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
            {"label": "laptop", "confidence": 0.9, "bbox": (210, 100, 400, 200)},
        ]
        relations = extract_relations(detections, near_threshold=50.0)
        near_rels = [r for r in relations if r.name == "NEAR"]
        self.assertTrue(len(near_rels) > 0)

    def test_contains_relation(self):
        detections = [
            {"label": "bowl", "confidence": 0.9, "bbox": (50, 50, 350, 350)},
            {"label": "apple", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
        ]
        relations = extract_relations(detections, containment_ratio=0.7)
        contains_rels = [r for r in relations if r.name == "CONTAINS"]
        self.assertTrue(any(r.src == "bowl_0" and r.dst == "apple_1" for r in contains_rels))

    def test_in_front_of_relation(self):
        """Object with lower depth is IN_FRONT_OF object with higher depth."""
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200), "depth": 0.2},
            {"label": "laptop", "confidence": 0.9, "bbox": (300, 100, 500, 200), "depth": 0.6},
        ]
        relations = extract_relations(detections, depth_threshold=0.15)
        front_rels = [r for r in relations if r.name == "IN_FRONT_OF"]
        self.assertTrue(any(r.src == "cup_0" and r.dst == "laptop_1" for r in front_rels))

    def test_behind_relation(self):
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200), "depth": 0.8},
            {"label": "laptop", "confidence": 0.9, "bbox": (300, 100, 500, 200), "depth": 0.3},
        ]
        relations = extract_relations(detections, depth_threshold=0.15)
        behind_rels = [r for r in relations if r.name == "BEHIND"]
        self.assertTrue(any(r.src == "cup_0" and r.dst == "laptop_1" for r in behind_rels))

    def test_no_depth_relations_without_depth(self):
        """Without depth estimates, no IN_FRONT_OF or BEHIND relations."""
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
            {"label": "laptop", "confidence": 0.9, "bbox": (300, 100, 500, 200)},
        ]
        relations = extract_relations(detections)
        depth_rels = [r for r in relations if r.name in ("IN_FRONT_OF", "BEHIND")]
        self.assertEqual(len(depth_rels), 0)


# ---------------------------------------------------------------------------
# Build world (with properties + depth)
# ---------------------------------------------------------------------------

class TestBuildWorld(unittest.TestCase):

    def _make_mock_model(self):
        mock_model = MagicMock()
        mock_model.names = {41: "cup", 60: "dining table"}
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [150.0, 80.0, 250.0, 150.0],
            [50.0, 150.0, 400.0, 300.0],
        ])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.90, 0.85])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([41.0, 60.0])
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        return mock_model

    def test_entities_have_physics_properties(self):
        ws = build_world("fake.jpg", model=self._make_mock_model(), use_properties=True)
        cup = ws.entities["cup_0"]
        self.assertTrue(cup.get("fragile"))
        self.assertIsInstance(cup.get("mass"), (int, float))
        table = ws.entities["dining table_1"]
        self.assertEqual(table.get("material"), "wood")

    def test_entities_have_depth(self):
        ws = build_world("fake.jpg", model=self._make_mock_model(), use_depth=True)
        cup = ws.entities["cup_0"]
        self.assertIn("depth", dict(cup.properties))

    def test_without_properties(self):
        ws = build_world("fake.jpg", model=self._make_mock_model(), use_properties=False)
        cup = ws.entities["cup_0"]
        self.assertNotIn("fragile", dict(cup.properties))

    def test_without_depth(self):
        ws = build_world("fake.jpg", model=self._make_mock_model(), use_depth=False)
        cup = ws.entities["cup_0"]
        self.assertNotIn("depth", dict(cup.properties))


# ---------------------------------------------------------------------------
# IoU tracking
# ---------------------------------------------------------------------------

class TestTrackObjects(unittest.TestCase):

    def test_same_object_matched(self):
        """Same object in slightly different position should match."""
        prev = [{"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)}]
        curr = [{"label": "cup", "confidence": 0.9, "bbox": (110, 105, 210, 205)}]
        matches = track_objects(prev, curr)
        self.assertEqual(matches[0], 0)

    def test_new_object_is_none(self):
        """Object with no match in previous frame maps to None."""
        prev = [{"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)}]
        curr = [
            {"label": "cup", "confidence": 0.9, "bbox": (105, 105, 205, 205)},
            {"label": "laptop", "confidence": 0.9, "bbox": (400, 400, 600, 500)},
        ]
        matches = track_objects(prev, curr)
        self.assertEqual(matches[0], 0)  # cup matched
        self.assertIsNone(matches[1])     # laptop is new

    def test_different_labels_not_matched(self):
        """Objects with different labels should not match even if overlapping."""
        prev = [{"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)}]
        curr = [{"label": "laptop", "confidence": 0.9, "bbox": (100, 100, 200, 200)}]
        matches = track_objects(prev, curr)
        self.assertIsNone(matches[0])

    def test_no_overlap_not_matched(self):
        prev = [{"label": "cup", "confidence": 0.9, "bbox": (0, 0, 50, 50)}]
        curr = [{"label": "cup", "confidence": 0.9, "bbox": (400, 400, 500, 500)}]
        matches = track_objects(prev, curr)
        self.assertIsNone(matches[0])

    def test_empty_frames(self):
        self.assertEqual(track_objects([], []), {})
        matches = track_objects([], [{"label": "cup", "confidence": 0.9, "bbox": (0, 0, 1, 1)}])
        self.assertIsNone(matches[0])


if __name__ == "__main__":
    unittest.main()
