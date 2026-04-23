"""Tests for the JavaExtractor."""

from __future__ import annotations

import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.java import JavaExtractor


def _parse(source: str):
    """Convenience: parse Java *source* and return the tree."""
    return TelosParser().parse(source.encode(), "java")


class TestExtractFunctions(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaExtractor()

    def test_method_declarations(self):
        src = (
            "public class Foo {\n"
            "    public String getName() { return name; }\n"
            "    public void setName(String n) { this.name = n; }\n"
            "}\n"
        )
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "Foo.java")
        names = [f["name"] for f in funcs]
        self.assertIn("getName", names)
        self.assertIn("setName", names)

    def test_method_kind(self):
        src = "public class A { void m() {} }\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "A.java")
        self.assertEqual(funcs[0]["kind"], "method")

    def test_method_line_numbers(self):
        src = "public class A {\n    void foo() {\n    }\n}\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "A.java")
        self.assertEqual(funcs[0]["line_start"], 2)

    def test_empty_class(self):
        src = "public class Empty {}\n"
        tree = _parse(src)
        funcs = self.extractor.extract_functions(tree, "Empty.java")
        self.assertEqual(funcs, [])


class TestExtractClasses(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaExtractor()

    def test_class_declaration(self):
        src = "public class Animal { String name; }\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "Animal.java")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "Animal")
        self.assertEqual(classes[0]["kind"], "class")

    def test_interface_declaration(self):
        src = "interface Shape { double area(); }\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "Shape.java")
        ifaces = [c for c in classes if c["kind"] == "interface"]
        self.assertEqual(len(ifaces), 1)
        self.assertEqual(ifaces[0]["name"], "Shape")

    def test_class_and_interface(self):
        src = (
            "public class Foo {}\n"
            "interface Bar {}\n"
        )
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "test.java")
        names = [c["name"] for c in classes]
        self.assertIn("Foo", names)
        self.assertIn("Bar", names)

    def test_class_file_path(self):
        src = "public class X {}\n"
        tree = _parse(src)
        classes = self.extractor.extract_classes(tree, "com/example/X.java")
        self.assertEqual(classes[0]["file_path"], "com/example/X.java")


class TestExtractImports(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaExtractor()

    def test_import_declarations(self):
        src = "import java.util.List;\nimport java.io.File;\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "Test.java")
        modules = [i["module"] for i in imports]
        self.assertIn("java.util.List", modules)
        self.assertIn("java.io.File", modules)

    def test_import_line_and_file(self):
        src = "import java.util.Map;\n"
        tree = _parse(src)
        imports = self.extractor.extract_imports(tree, "App.java")
        self.assertEqual(imports[0]["line"], 1)
        self.assertEqual(imports[0]["file_path"], "App.java")


class TestExtractCalls(unittest.TestCase):
    def setUp(self):
        self.extractor = JavaExtractor()

    def test_method_invocations(self):
        src = (
            "public class Main {\n"
            "    public void run() {\n"
            '        System.out.println("hello");\n'
            "        helper();\n"
            "    }\n"
            "}\n"
        )
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "Main.java")
        names = [c["name"] for c in calls]
        self.assertIn("println", names)
        self.assertIn("helper", names)

    def test_call_full_name_with_object(self):
        src = (
            "public class Main {\n"
            '    public void run() { obj.doWork(); }\n'
            "}\n"
        )
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "Main.java")
        self.assertTrue(any(c["full_name"] == "obj.doWork" for c in calls))

    def test_calls_empty(self):
        src = "public class Empty {}\n"
        tree = _parse(src)
        calls = self.extractor.extract_calls(tree, "Empty.java")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
