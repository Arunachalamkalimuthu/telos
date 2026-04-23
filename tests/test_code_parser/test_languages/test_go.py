"""Tests for the GoExtractor."""

from __future__ import annotations

import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.go import GoExtractor


def _parse(source: str):
    """Convenience: parse Go *source* and return the tree."""
    return TelosParser().parse(source.encode(), "go")


class TestExtractFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = GoExtractor()

    def test_function_declarations(self):
        src = (
            "package main\n"
            "func hello(name string) string { return name }\n"
            "func main() {}\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "main.go")
        names = [f["name"] for f in funcs]
        self.assertIn("hello", names)
        self.assertIn("main", names)

    def test_function_kind(self):
        src = "package main\nfunc foo() {}\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "main.go")
        self.assertEqual(funcs[0]["kind"], "function")

    def test_method_declarations(self):
        src = (
            "package main\n"
            "type Point struct { X int; Y int }\n"
            "func (p *Point) Move(dx int) { p.X += dx }\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "main.go")
        methods = [f for f in funcs if f["kind"] == "method"]
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0]["name"], "Move")

    def test_empty_source(self):
        tree = _parse("package main\n")
        funcs = self.extractor.extract_functions(tree, "main.go")
        self.assertEqual(funcs, [])


class TestExtractClasses(unittest.TestCase):
    def setUp(self):
        self.extractor = GoExtractor()

    def test_struct_declaration(self):
        src = (
            "package main\n"
            "type Point struct {\n"
            "    X int\n"
            "    Y int\n"
            "}\n"
        )
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "main.go")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "Point")
        self.assertEqual(classes[0]["kind"], "struct")

    def test_struct_file_path(self):
        src = "package main\ntype Foo struct {}\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "pkg/model.go")
        self.assertEqual(classes[0]["file_path"], "pkg/model.go")


class TestExtractImports(unittest.TestCase):
    def setUp(self):
        self.extractor = GoExtractor()

    def test_grouped_imports(self):
        src = (
            'package main\n'
            'import (\n'
            '    "fmt"\n'
            '    "os"\n'
            ')\n'
        )
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "main.go")
        modules = [i["module"] for i in imports]
        self.assertIn("fmt", modules)
        self.assertIn("os", modules)

    def test_import_line_and_file(self):
        src = 'package main\nimport "fmt"\n'
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "main.go")
        self.assertEqual(imports[0]["module"], "fmt")
        self.assertEqual(imports[0]["file_path"], "main.go")


class TestExtractCalls(unittest.TestCase):
    def setUp(self):
        self.extractor = GoExtractor()

    def test_simple_and_selector_calls(self):
        src = (
            'package main\n'
            'func main() {\n'
            '    fmt.Println("hello")\n'
            '    hello("world")\n'
            '}\n'
        )
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "main.go")
        names = [c["name"] for c in calls]
        self.assertIn("Println", names)
        self.assertIn("hello", names)

    def test_call_full_name(self):
        src = 'package main\nfunc main() { fmt.Println("hi") }\n'
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "main.go")
        self.assertTrue(any(c["full_name"] == "fmt.Println" for c in calls))

    def test_calls_empty_source(self):
        tree = _parse("package main\n")
        calls = self.extractor.extract_calls(tree, "main.go")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
