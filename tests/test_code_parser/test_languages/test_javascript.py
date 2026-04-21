"""Tests for the JavaScriptExtractor."""

from __future__ import annotations

import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.javascript import JavaScriptExtractor


def _parse(source: str):
    """Convenience: parse JavaScript *source* and return the tree."""
    return TelosParser().parse(source.encode(), "javascript")


class TestExtractFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaScriptExtractor()

    def test_extract_functions(self):
        src = "function hello(name) { return name; }\nfunction world() {}"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.js")
        names = [f["name"] for f in funcs]
        self.assertIn("hello", names)
        self.assertIn("world", names)

    def test_extract_functions_kind(self):
        src = "function foo() {}"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.js")
        self.assertEqual(funcs[0]["kind"], "function")

    def test_extract_functions_file_path(self):
        src = "function foo() {}"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "my/module.js")
        self.assertEqual(funcs[0]["file_path"], "my/module.js")

    def test_extract_functions_line_numbers(self):
        src = "function foo() {\n  return 1;\n}"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.js")
        self.assertEqual(funcs[0]["line_start"], 1)
        self.assertGreaterEqual(funcs[0]["line_end"], 1)

    def test_extract_functions_empty_source(self):
        tree = _parse("")
        funcs = self.extractor.extract_functions(tree, "test.js")
        self.assertEqual(funcs, [])


class TestExtractArrowFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaScriptExtractor()

    def test_extract_arrow_functions(self):
        src = "const greet = (name) => { return name; };"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.js")
        names = [f["name"] for f in funcs]
        self.assertIn("greet", names)

    def test_extract_arrow_function_let(self):
        src = "let add = (a, b) => a + b;"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.js")
        names = [f["name"] for f in funcs]
        self.assertIn("add", names)

    def test_extract_arrow_function_kind(self):
        src = "const greet = (name) => { return name; };"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "test.js")
        self.assertEqual(funcs[0]["kind"], "function")


class TestExtractClasses(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaScriptExtractor()

    def test_extract_classes(self):
        src = "class MyClass extends Base { constructor() {} }"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.js")
        self.assertEqual(len(classes), 1)
        cls = classes[0]
        self.assertEqual(cls["name"], "MyClass")
        self.assertEqual(cls["kind"], "class")
        self.assertIn("Base", cls["bases"])

    def test_extract_classes_no_bases(self):
        src = "class Plain { constructor() {} }"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.js")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["bases"], [])

    def test_extract_classes_file_path(self):
        src = "class Foo {}"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "pkg/models.js")
        self.assertEqual(classes[0]["file_path"], "pkg/models.js")


class TestExtractImports(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaScriptExtractor()

    def test_extract_imports(self):
        src = "import { foo } from 'bar';"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.js")
        modules = [i["module"] for i in imports]
        self.assertIn("bar", modules)

    def test_extract_imports_double_quotes(self):
        src = 'import React from "react";'
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.js")
        modules = [i["module"] for i in imports]
        self.assertIn("react", modules)

    def test_extract_imports_line_number(self):
        src = "import { foo } from 'bar';"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.js")
        self.assertEqual(imports[0]["line"], 1)

    def test_extract_imports_file_path(self):
        src = "import { foo } from 'bar';"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "pkg/index.js")
        self.assertEqual(imports[0]["file_path"], "pkg/index.js")

    def test_extract_imports_multiple(self):
        src = "import { a } from 'mod-a';\nimport { b } from 'mod-b';"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "test.js")
        modules = [i["module"] for i in imports]
        self.assertIn("mod-a", modules)
        self.assertIn("mod-b", modules)


class TestExtractCalls(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaScriptExtractor()

    def test_extract_calls(self):
        src = "function main() { hello(); console.log('hi'); }"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.js")
        names = [c["name"] for c in calls]
        self.assertIn("hello", names)
        self.assertIn("log", names)

    def test_extract_calls_full_name_member(self):
        src = "console.log('hi');"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.js")
        self.assertTrue(any(c["full_name"] == "console.log" for c in calls))

    def test_extract_calls_file_path(self):
        src = "foo();"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "pkg/main.js")
        self.assertEqual(calls[0]["file_path"], "pkg/main.js")

    def test_extract_calls_line_number(self):
        src = "const x = 1;\nfoo();\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "test.js")
        foo_calls = [c for c in calls if c["name"] == "foo"]
        self.assertEqual(foo_calls[0]["line"], 2)

    def test_extract_calls_empty_source(self):
        tree = _parse("")
        calls = self.extractor.extract_calls(tree, "test.js")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
