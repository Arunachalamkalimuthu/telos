"""Tests for the RustExtractor."""

from __future__ import annotations

import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.rust import RustExtractor


def _parse(source: str):
    """Convenience: parse Rust *source* and return the tree."""
    return TelosParser().parse(source.encode(), "rust")


class TestExtractFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = RustExtractor()

    def test_function_items(self):
        src = (
            "fn hello(name: &str) -> String { name.to_string() }\n"
            "fn main() {}\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "main.rs")
        names = [f["name"] for f in funcs]
        self.assertIn("hello", names)
        self.assertIn("main", names)

    def test_function_kind(self):
        src = "fn foo() {}\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "main.rs")
        self.assertEqual(funcs[0]["kind"], "function")

    def test_impl_method(self):
        src = (
            "struct Point { x: f64 }\n"
            "impl Point {\n"
            "    fn new(x: f64) -> Self { Point { x } }\n"
            "}\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "main.rs")
        names = [f["name"] for f in funcs]
        self.assertIn("new", names)

    def test_empty_source(self):
        tree = _parse("")
        funcs = self.extractor.extract_functions(tree, "main.rs")
        self.assertEqual(funcs, [])


class TestExtractClasses(unittest.TestCase):
    def setUp(self):
        self.extractor = RustExtractor()

    def test_struct_item(self):
        src = "struct Point {\n    x: f64,\n    y: f64,\n}\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "main.rs")
        structs = [c for c in classes if c["kind"] == "struct"]
        self.assertEqual(len(structs), 1)
        self.assertEqual(structs[0]["name"], "Point")

    def test_enum_item(self):
        src = "enum Color {\n    Red,\n    Green,\n    Blue,\n}\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "main.rs")
        enums = [c for c in classes if c["kind"] == "enum"]
        self.assertEqual(len(enums), 1)
        self.assertEqual(enums[0]["name"], "Color")

    def test_impl_item(self):
        src = (
            "struct Foo {}\n"
            "impl Foo {\n"
            "    fn bar() {}\n"
            "}\n"
        )
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "main.rs")
        impls = [c for c in classes if c["kind"] == "impl"]
        self.assertEqual(len(impls), 1)
        self.assertEqual(impls[0]["name"], "Foo")

    def test_struct_file_path(self):
        src = "struct X {}\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "src/model.rs")
        self.assertEqual(classes[0]["file_path"], "src/model.rs")


class TestExtractImports(unittest.TestCase):
    def setUp(self):
        self.extractor = RustExtractor()

    def test_use_declarations(self):
        src = "use std::io;\nuse std::collections::HashMap;\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "main.rs")
        modules = [i["module"] for i in imports]
        self.assertIn("std::io", modules)
        self.assertIn("std::collections::HashMap", modules)

    def test_import_line_and_file(self):
        src = "use std::io;\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "lib.rs")
        self.assertEqual(imports[0]["line"], 1)
        self.assertEqual(imports[0]["file_path"], "lib.rs")


class TestExtractCalls(unittest.TestCase):
    def setUp(self):
        self.extractor = RustExtractor()

    def test_simple_and_scoped_calls(self):
        src = (
            "fn main() {\n"
            "    let p = Point::new(1.0, 2.0);\n"
            '    hello("world");\n'
            "}\n"
        )
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "main.rs")
        names = [c["name"] for c in calls]
        self.assertIn("new", names)
        self.assertIn("hello", names)

    def test_call_full_name(self):
        src = "fn main() { Point::new(1.0, 2.0); }\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "main.rs")
        self.assertTrue(any(c["full_name"] == "Point::new" for c in calls))

    def test_calls_empty(self):
        tree = _parse("")
        calls = self.extractor.extract_calls(tree, "main.rs")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
