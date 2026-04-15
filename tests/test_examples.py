import unittest
from io import StringIO
from contextlib import redirect_stdout
import importlib


class TestCoffeeCupExample(unittest.TestCase):
    def test_coffee_cup_runs_and_includes_causal_chain(self):
        buf = StringIO()
        with redirect_stdout(buf):
            mod = importlib.import_module("examples.coffee_cup")
            mod.run()
        out = buf.getvalue()
        self.assertIn("cup.contents_escape", out)
        self.assertIn("seal", out.lower())
        self.assertIn("counterfactual", out.lower())


if __name__ == "__main__":
    unittest.main()
