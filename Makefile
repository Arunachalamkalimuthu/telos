.PHONY: test demo clean

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	@PYTHONPATH=src python3 -m examples.coffee_cup
	@echo
	@PYTHONPATH=src python3 -m examples.child_road
	@echo
	@PYTHONPATH=src python3 -m examples.salt_request
	@echo
	@PYTHONPATH=src python3 -m examples.novel_entity

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
