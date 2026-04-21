"""Tests for the multi-language tree-sitter parser orchestrator."""

import unittest

from telos.code_parser.parser import TelosParser


class TestTelosParserDetectLanguage(unittest.TestCase):
    def test_detect_language_python(self):
        self.assertEqual(TelosParser.detect_language("foo.py"), "python")

    def test_detect_language_javascript(self):
        self.assertEqual(TelosParser.detect_language("foo.js"), "javascript")

    def test_detect_language_jsx(self):
        self.assertEqual(TelosParser.detect_language("foo.jsx"), "javascript")

    def test_detect_language_typescript(self):
        self.assertEqual(TelosParser.detect_language("foo.ts"), "typescript")

    def test_detect_language_tsx(self):
        self.assertEqual(TelosParser.detect_language("foo.tsx"), "tsx")

    def test_detect_language_go(self):
        self.assertEqual(TelosParser.detect_language("foo.go"), "go")

    def test_detect_language_java(self):
        self.assertEqual(TelosParser.detect_language("Foo.java"), "java")

    def test_detect_language_rust(self):
        self.assertEqual(TelosParser.detect_language("foo.rs"), "rust")

    def test_detect_language_unknown(self):
        self.assertIsNone(TelosParser.detect_language("foo.rb"))

    def test_detect_language_no_extension(self):
        self.assertIsNone(TelosParser.detect_language("Makefile"))


class TestTelosParserSupportedLanguages(unittest.TestCase):
    def test_supported_languages(self):
        langs = TelosParser.supported_languages()
        expected = {"python", "javascript", "tsx", "typescript", "go", "java", "rust"}
        self.assertEqual(set(langs), expected)
        self.assertEqual(len(langs), 7)


class TestTelosParserParse(unittest.TestCase):
    def setUp(self):
        self.parser = TelosParser()

    def test_parse_python_source(self):
        tree = self.parser.parse(b"def hello(): pass", "python")
        self.assertEqual(tree.root_node.type, "module")

    def test_parse_javascript_source(self):
        tree = self.parser.parse(b"function hello() {}", "javascript")
        self.assertEqual(tree.root_node.type, "program")

    def test_parse_typescript_source(self):
        tree = self.parser.parse(b"function hello(): void {}", "typescript")
        self.assertEqual(tree.root_node.type, "program")

    def test_parse_tsx_source(self):
        tree = self.parser.parse(b"const el = <div />;", "tsx")
        self.assertEqual(tree.root_node.type, "program")

    def test_parse_go_source(self):
        tree = self.parser.parse(b"package main", "go")
        self.assertEqual(tree.root_node.type, "source_file")

    def test_parse_java_source(self):
        tree = self.parser.parse(b"class Foo {}", "java")
        self.assertEqual(tree.root_node.type, "program")

    def test_parse_rust_source(self):
        tree = self.parser.parse(b"fn main() {}", "rust")
        self.assertEqual(tree.root_node.type, "source_file")

    def test_parse_invalid_language_raises(self):
        with self.assertRaises(KeyError):
            self.parser.parse(b"", "cobol")


if __name__ == "__main__":
    unittest.main()
