"""Tests for the perception module (YOLOv8-nano → WorldState)."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from telos.perception import build_world, detect_objects, extract_relations
from telos.world import Relation


class TestDetectObjects(unittest.TestCase):
    """Mock YOLO and verify detect_objects returns the expected detection list."""

    def test_single_detection(self):
        # Build a mock YOLO model that returns one detection: cls=41 (cup), conf=0.95
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
        self.assertEqual(detections[0]["bbox"], (100.0, 200.0, 300.0, 400.0))

    def test_below_threshold_filtered(self):
        mock_model = MagicMock()
        mock_model.names = {41: "cup"}

        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [[10.0, 20.0, 30.0, 40.0]]
        )
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.1])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([41.0])

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detections = detect_objects("fake.jpg", model=mock_model, confidence_threshold=0.3)
        self.assertEqual(len(detections), 0)


class TestExtractRelations(unittest.TestCase):
    """Verify spatial relation extraction from bounding boxes."""

    def test_on_relation(self):
        # cup on top of table: cup's bottom edge (150) is at table's top edge (150)
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (150, 80, 250, 150)},
            {"label": "table", "confidence": 0.9, "bbox": (50, 150, 400, 300)},
        ]
        relations = extract_relations(detections)
        on_rels = [r for r in relations if r.name == "ON"]
        self.assertTrue(
            any(r.src == "cup_0" and r.dst == "table_1" for r in on_rels),
            f"Expected ON(cup_0, table_1) in {on_rels}",
        )

    def test_near_relation(self):
        # cup and laptop close together (gap of 10px horizontally)
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
            {"label": "laptop", "confidence": 0.9, "bbox": (210, 100, 400, 200)},
        ]
        relations = extract_relations(detections, near_threshold=50.0)
        near_rels = [r for r in relations if r.name == "NEAR"]
        self.assertTrue(
            any(
                (r.src == "cup_0" and r.dst == "laptop_1")
                or (r.src == "laptop_1" and r.dst == "cup_0")
                for r in near_rels
            ),
            f"Expected NEAR relation between cup_0 and laptop_1 in {near_rels}",
        )

    def test_contains_relation(self):
        # bowl contains apple (apple fully inside bowl)
        detections = [
            {"label": "bowl", "confidence": 0.9, "bbox": (50, 50, 350, 350)},
            {"label": "apple", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
        ]
        relations = extract_relations(detections, containment_ratio=0.7)
        contains_rels = [r for r in relations if r.name == "CONTAINS"]
        self.assertTrue(
            any(r.src == "bowl_0" and r.dst == "apple_1" for r in contains_rels),
            f"Expected CONTAINS(bowl_0, apple_1) in {contains_rels}",
        )


class TestBuildWorld(unittest.TestCase):
    """Verify build_world returns a WorldState with correct entities."""

    def test_build_world_entities(self):
        mock_model = MagicMock()
        mock_model.names = {41: "cup", 60: "dining table"}

        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [
                [150.0, 80.0, 250.0, 150.0],
                [50.0, 150.0, 400.0, 300.0],
            ]
        )
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.90, 0.85])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([41.0, 60.0])

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        ws = build_world("fake.jpg", model=mock_model)

        self.assertIn("cup_0", ws.entities)
        self.assertIn("dining table_1", ws.entities)
        self.assertEqual(ws.entities["cup_0"].type, "cup")
        self.assertEqual(ws.entities["dining table_1"].type, "dining table")
        self.assertAlmostEqual(ws.entities["cup_0"].get("confidence"), 0.90)
        self.assertAlmostEqual(ws.entities["dining table_1"].get("confidence"), 0.85)


if __name__ == "__main__":
    unittest.main()
