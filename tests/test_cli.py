"""Tests for the telos CLI."""

import os
import tempfile

from typer.testing import CliRunner

from telos.cli import app

runner = CliRunner()


def _write_sample_py(directory: str) -> str:
    """Write a minimal Python file so scan_directory finds something."""
    fp = os.path.join(directory, "sample.py")
    with open(fp, "w") as f:
        f.write(
            "def greet(name):\n"
            "    return f'hello {name}'\n\n"
            "def main():\n"
            "    greet('world')\n"
        )
    return fp


def test_init_creates_db():
    with tempfile.TemporaryDirectory() as tmp:
        _write_sample_py(tmp)
        result = runner.invoke(app, ["init", tmp])
        assert result.exit_code == 0, result.output
        db_path = os.path.join(tmp, ".telos", "graph.db")
        assert os.path.exists(db_path)


def test_init_shows_summary():
    with tempfile.TemporaryDirectory() as tmp:
        _write_sample_py(tmp)
        result = runner.invoke(app, ["init", tmp])
        assert result.exit_code == 0, result.output
        assert "Scanning" in result.output


def test_info_after_init():
    with tempfile.TemporaryDirectory() as tmp:
        _write_sample_py(tmp)
        runner.invoke(app, ["init", tmp])
        result = runner.invoke(app, ["info", "--path", tmp])
        assert result.exit_code == 0, result.output


def test_hotspots_after_init():
    with tempfile.TemporaryDirectory() as tmp:
        _write_sample_py(tmp)
        runner.invoke(app, ["init", tmp])
        result = runner.invoke(app, ["hotspots", "--path", tmp])
        assert result.exit_code == 0, result.output


def test_impact_after_init():
    with tempfile.TemporaryDirectory() as tmp:
        _write_sample_py(tmp)
        runner.invoke(app, ["init", tmp])
        # Use a node id that the scanner would create.
        result = runner.invoke(app, ["impact", "sample.py:greet", "--path", tmp])
        assert result.exit_code == 0, result.output
