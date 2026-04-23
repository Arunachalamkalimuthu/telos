"""Tests for the TypeScriptExtractor."""

from __future__ import annotations

import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.typescript import TypeScriptExtractor


def _parse(source: str):
    """Convenience: parse TypeScript *source* and return the tree."""
    return TelosParser().parse(source.encode(), "typescript")


class TestExtractFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = TypeScriptExtractor()

    def test_named_functions(self):
        src = "function greet(name: string): string { return name; }\nfunction bye(): void {}"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.ts")
        names = [f["name"] for f in funcs]
        self.assertIn("greet", names)
        self.assertIn("bye", names)

    def test_arrow_functions(self):
        src = "const add = (a: number, b: number): number => a + b;"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.ts")
        names = [f["name"] for f in funcs]
        self.assertIn("add", names)

    def test_function_kind_and_line(self):
        src = "function foo(): void {}"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.ts")
        self.assertEqual(funcs[0]["kind"], "function")
        self.assertEqual(funcs[0]["line_start"], 1)

    def test_empty_source(self):
        tree = _parse("")
        funcs = self.extractor.extract_functions(tree, "test.ts")
        self.assertEqual(funcs, [])


class TestExtractClasses(unittest.TestCase):
    def setUp(self):
        self.extractor = TypeScriptExtractor()

    def test_class_declaration(self):
        src = "class Animal extends Base { name: string; }"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.ts")
        cls_names = [c["name"] for c in classes]
        self.assertIn("Animal", cls_names)
        animal = [c for c in classes if c["name"] == "Animal"][0]
        self.assertEqual(animal["kind"], "class")
        self.assertIn("Base", animal["bases"])

    def test_interface_declaration(self):
        src = "interface Shape { area(): number; }"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.ts")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "Shape")
        self.assertEqual(classes[0]["kind"], "interface")

    def test_type_alias_declaration(self):
        src = "type ID = string | number;"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.ts")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "ID")
        self.assertEqual(classes[0]["kind"], "type_alias")

    def test_mixed_class_interface_type(self):
        src = (
            "class Foo {}\n"
            "interface Bar { x: number; }\n"
            "type Baz = string;\n"
        )
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.ts")
        names = [c["name"] for c in classes]
        self.assertIn("Foo", names)
        self.assertIn("Bar", names)
        self.assertIn("Baz", names)


class TestExtractImports(unittest.TestCase):
    def setUp(self):
        self.extractor = TypeScriptExtractor()

    def test_import_statement(self):
        src = "import { foo } from 'bar';"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.ts")
        modules = [i["module"] for i in imports]
        self.assertIn("bar", modules)

    def test_import_line_and_file(self):
        src = "import { x } from 'mod';"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "app/index.ts")
        self.assertEqual(imports[0]["line"], 1)
        self.assertEqual(imports[0]["file_path"], "app/index.ts")


class TestExtractCalls(unittest.TestCase):
    def setUp(self):
        self.extractor = TypeScriptExtractor()

    def test_simple_and_member_calls(self):
        src = "greet('world');\nconsole.log('hi');"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.ts")
        names = [c["name"] for c in calls]
        self.assertIn("greet", names)
        self.assertIn("log", names)

    def test_call_full_name(self):
        src = "console.log('hi');"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.ts")
        self.assertTrue(any(c["full_name"] == "console.log" for c in calls))


if __name__ == "__main__":
    unittest.main()
