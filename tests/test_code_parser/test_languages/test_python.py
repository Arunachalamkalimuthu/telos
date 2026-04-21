"""Tests for the PythonExtractor."""

from __future__ import annotations

import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.python import PythonExtractor


def _parse(source: str):
    """Convenience: parse Python *source* and return the tree."""
    return TelosParser().parse(source.encode(), "python")


class TestExtractFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonExtractor()

    def test_extract_functions(self):
        src = "def hello(name):\n    pass\n\ndef world():\n    pass\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.py")
        names = [f["name"] for f in funcs]
        self.assertIn("hello", names)
        self.assertIn("world", names)
        self.assertEqual(len(funcs), 2)

    def test_extract_functions_kind(self):
        tree = _parse("def foo(): pass\n")
        funcs = self.extractor.extract_functions(tree, "test.py")
        self.assertEqual(funcs[0]["kind"], "function")

    def test_extract_functions_file_path(self):
        tree = _parse("def foo(): pass\n")
        funcs = self.extractor.extract_functions(tree, "my/module.py")
        self.assertEqual(funcs[0]["file_path"], "my/module.py")

    def test_extract_functions_line_numbers(self):
        src = "def foo():\n    pass\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.py")
        self.assertEqual(funcs[0]["line_start"], 1)
        self.assertGreaterEqual(funcs[0]["line_end"], 1)

    def test_extract_functions_empty_source(self):
        tree = _parse("")
        funcs = self.extractor.extract_functions(tree, "test.py")
        self.assertEqual(funcs, [])


class TestExtractClasses(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonExtractor()

    def test_extract_classes(self):
        src = "class MyClass(Base):\n    def method(self):\n        pass\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.py")
        self.assertEqual(len(classes), 1)
        cls = classes[0]
        self.assertEqual(cls["name"], "MyClass")
        self.assertEqual(cls["kind"], "class")
        self.assertIn("Base", cls["bases"])

    def test_extract_classes_multiple_bases(self):
        src = "class Child(ParentA, ParentB):\n    pass\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.py")
        self.assertEqual(len(classes), 1)
        bases = classes[0]["bases"]
        self.assertIn("ParentA", bases)
        self.assertIn("ParentB", bases)

    def test_extract_classes_no_bases(self):
        src = "class Plain:\n    pass\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.py")
        self.assertEqual(classes[0]["bases"], [])

    def test_extract_classes_file_path(self):
        src = "class Foo:\n    pass\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "pkg/models.py")
        self.assertEqual(classes[0]["file_path"], "pkg/models.py")


class TestExtractImports(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonExtractor()

    def test_extract_imports(self):
        src = "import os\nfrom pathlib import Path\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.py")
        modules = [i["module"] for i in imports]
        self.assertIn("os", modules)
        self.assertIn("pathlib", modules)

    def test_extract_imports_names(self):
        src = "from pathlib import Path\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.py")
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["module"], "pathlib")
        self.assertIn("Path", imports[0]["names"])

    def test_extract_imports_plain_module_names_is_empty(self):
        src = "import os\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.py")
        self.assertEqual(imports[0]["names"], [])

    def test_extract_imports_line_number(self):
        src = "import os\nfrom pathlib import Path\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.py")
        by_module = {i["module"]: i for i in imports}
        self.assertEqual(by_module["os"]["line"], 1)
        self.assertEqual(by_module["pathlib"]["line"], 2)

    def test_extract_imports_file_path(self):
        src = "import sys\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "pkg/utils.py")
        self.assertEqual(imports[0]["file_path"], "pkg/utils.py")

    def test_extract_imports_multiple_names(self):
        src = "from os.path import join, exists\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.py")
        self.assertEqual(imports[0]["module"], "os.path")
        self.assertIn("join", imports[0]["names"])
        self.assertIn("exists", imports[0]["names"])


class TestExtractCalls(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonExtractor()

    def test_extract_calls(self):
        src = "def main():\n    hello()\n    obj.method()\n    print('hi')\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.py")
        names = [c["name"] for c in calls]
        self.assertIn("hello", names)
        self.assertIn("method", names)
        self.assertIn("print", names)

    def test_extract_calls_full_name_attribute(self):
        src = "obj.method()\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.py")
        self.assertTrue(any(c["full_name"] == "obj.method" for c in calls))

    def test_extract_calls_file_path(self):
        src = "foo()\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "pkg/main.py")
        self.assertEqual(calls[0]["file_path"], "pkg/main.py")

    def test_extract_calls_line_number(self):
        src = "x = 1\nfoo()\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.py")
        foo_calls = [c for c in calls if c["name"] == "foo"]
        self.assertEqual(foo_calls[0]["line"], 2)

    def test_extract_calls_empty_source(self):
        tree = _parse("")
        calls = self.extractor.extract_calls(tree, "test.py")
        self.assertEqual(calls, [])


class TestExtractMethodsInsideClass(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonExtractor()

    def test_extract_methods_inside_class(self):
        src = (
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        pass\n"
            "    def method_b(self):\n"
            "        pass\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.py")
        names = [f["name"] for f in funcs]
        self.assertIn("method_a", names)
        self.assertIn("method_b", names)

    def test_extract_methods_and_top_level_functions(self):
        src = (
            "def standalone():\n"
            "    pass\n"
            "\n"
            "class Foo:\n"
            "    def inside(self):\n"
            "        pass\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.py")
        names = [f["name"] for f in funcs]
        self.assertIn("standalone", names)
        self.assertIn("inside", names)
        self.assertEqual(len(funcs), 2)


class TestExtractDecoratedFunction(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonExtractor()

    def test_extract_decorated_function(self):
        src = "@app.route('/api')\ndef handler():\n    pass\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.py")
        names = [f["name"] for f in funcs]
        self.assertIn("handler", names)

    def test_extract_multiple_decorators(self):
        src = (
            "@decorator_one\n"
            "@decorator_two\n"
            "def my_func():\n"
            "    pass\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.py")
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0]["name"], "my_func")


if __name__ == "__main__":
    unittest.main()
