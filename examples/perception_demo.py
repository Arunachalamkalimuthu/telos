"""Perception pipeline: image → YOLOv8-nano detections → WorldState → causal graph."""

import os
import sys

from telos import CAWAAgent
from telos.perception import build_world

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_SAMPLE_IMAGE = os.path.join(_ASSETS_DIR, "sample_scene.png")


def _create_sample_image(path: str) -> None:
    """Create a synthetic table+cup scene image.

    Tries PIL/Pillow first; falls back to a minimal 1×1 PNG if unavailable.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        # Table surface — wide brown rectangle in the lower half
        draw.rectangle([50, 320, 590, 380], fill=(139, 90, 43))
        # Cup — tall rectangle sitting on the table
        draw.rectangle([270, 220, 360, 320], fill=(220, 100, 60))
        img.save(path)
        print(f"  Created synthetic scene image (PIL): {path}")
    except ImportError:
        # Minimal valid 1×1 white PNG (67 bytes, no external deps)
        _MINIMAL_PNG = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with open(path, "wb") as fh:
            fh.write(_MINIMAL_PNG)
        print(f"  Created minimal fallback PNG (no PIL): {path}")


def run() -> None:
    print("=== Example: perception pipeline (YOLOv8-nano → WorldState) ===")
    print()

    # --- resolve image path ---
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using provided image: {image_path}")
    else:
        image_path = _SAMPLE_IMAGE
        print("No image path supplied; generating sample scene …")
        _create_sample_image(image_path)

    print()

    # --- run perception pipeline ---
    print("Running build_world() …")
    world = build_world(image_path)
    print()

    # --- detected objects ---
    print("Detected objects:")
    if world.entities:
        for eid, entity in world.entities.items():
            conf = entity.get("confidence")
            bbox = entity.get("bbox")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
            print(f"  {eid}  confidence={conf_str}  bbox={bbox}")
    else:
        print("  (none detected above confidence threshold)")
    print()

    # --- extracted relations ---
    print("Extracted spatial relations:")
    if world.relations:
        for rel in world.relations:
            print(f"  {rel.type}({rel.subject}, {rel.object})")
    else:
        print("  (none)")
    print()

    # --- causal graph ---
    agent = CAWAAgent()
    agent.perceive(world)
    graph = agent.build_causal_graph()

    variables = graph.variables()
    edges = graph.all_edges()

    print("Causal graph:")
    print(f"  Variables  : {len(variables)}")
    print(f"  Causal edges: {len(edges)}")

    if variables:
        print("  Variable list:")
        for var in sorted(variables):
            print(f"    {var}")
        print()
        state = graph.propagate()
        print("  Propagated state:")
        for var in sorted(state):
            print(f"    {var} = {state[var]}")
    else:
        print()
        print("Note: raw YOLO detections carry only spatial/visual properties")
        print("(confidence, bbox).  The physics primitives (gravity, liquid_damage,")
        print("impact, containment) require domain properties such as 'mass',")
        print("'orientation', or 'contains' to fire.  No causal edges were inferred")
        print("from this scene; a richer ontology mapping is needed to derive physics.")

    print()


if __name__ == "__main__":
    run()
