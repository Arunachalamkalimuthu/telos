"""Dinner table "Can you pass me the salt?": theory of mind + social inference."""

from cawa import AgentMind, Entity, Relation, WorldState


def scene():
    salt = Entity(id="salt", type="salt_shaker", properties={"position": "centre"})
    cast = Entity(id="cast", type="cast", properties={"on_arm": True})
    asker_belief = WorldState(
        entities={"salt": salt, "cast": cast},
        relations=(
            Relation("BEYOND_REACH_OF", "salt", "asker"),
            Relation("WORN_BY", "cast", "asker"),
        ),
    )
    asker = AgentMind(
        id="asker",
        beliefs=asker_belief,
        goals=({"type": "have", "target": "salt"},),
        capabilities=frozenset({"visual", "auditory", "verbal"}),
        actions=("ask_for_help",),
    )
    return asker, asker_belief


def run():
    print("=== Example: 'Can you pass me the salt?' ===")
    asker, world = scene()
    utterance = "Can you pass me the salt?"
    print(f"Utterance: {utterance!r}")
    print()

    # Reconstruct the asker's reasoning explicitly.
    print("Reading the asker's mind (theory of mind):")
    print(f"  asker.beliefs say: salt is beyond their reach (wrist in cast).")
    print(f"  asker.goals say:   they want salt.")
    print(f"  asker.capabilities include: verbal communication.")
    print(f"  asker.actions include: ask_for_help (the utterance we just heard).")
    print()

    print("Inferred communicative intent:")
    print(f"  surface form = capability question ('can you...?')")
    print(f"  but conditioned on asker's beliefs, literal reading makes no sense")
    print(f"  (they are not curious about my physical abilities; they cannot reach the salt)")
    print(f"  → interpret as a request: bring the salt within their reach.")
    print()

    print("Action planning with constraints:")
    print(f"  asker has a cast on their arm → grip is impaired.")
    print(f"  → place salt on their unaffected side, within easy reach.")
    print(f"  → if the shaker has a screw cap they cannot twist, loosen it first.")
    print()

    print("Chosen action: pass_salt_within_reach_and_pre_loosen_cap")


if __name__ == "__main__":
    run()
