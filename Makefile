.PHONY: test demo install clean

PYTHON ?= python3.11

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

demo:
	@PYTHONPATH=src $(PYTHON) -m examples.coffee_cup
	@echo
	@PYTHONPATH=src $(PYTHON) -m examples.child_road
	@echo
	@PYTHONPATH=src $(PYTHON) -m examples.salt_request
	@echo
	@PYTHONPATH=src $(PYTHON) -m examples.novel_entity
	@echo
	@PYTHONPATH=src $(PYTHON) -m examples.learned_structure
	@echo
	@PYTHONPATH=src $(PYTHON) -m examples.nlu_demo

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m spacy download en_core_web_sm

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
