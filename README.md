# CAWA — Causal Active World Architecture

A Python reference implementation of the architecture described in the Medium article *Beyond Token Prediction: How LLMs, JEPA, and CAWA See the World Differently*.

**This is not AGI.** This is a closed-domain proof that causal graphs, physics primitives, theory of mind, and active inference compose cleanly into a working agent. See `docs/architecture.md` for an honest accounting of what this does and does not demonstrate.

## Quickstart

```bash
make test      # run all tests
make demo      # run all four example scenarios
```

## Examples

- `examples/coffee_cup.py` — physics + counterfactuals
- `examples/child_road.py` — theory of mind + intervention planning
- `examples/salt_request.py` — theory of mind + social inference
- `examples/novel_entity.py` — honest uncertainty with unknown entities

## Requirements

Python 3.10+. No runtime dependencies.
