"""SQLite-backed storage for code dependency graphs."""

import os
import sqlite3


class GraphStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id        TEXT PRIMARY KEY,
                file_path TEXT,
                name      TEXT,
                kind      TEXT,
                language  TEXT,
                line_start INTEGER DEFAULT 0,
                line_end   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS edges (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                source    TEXT,
                target    TEXT,
                kind      TEXT,
                weight    REAL DEFAULT 1.0,
                file_path TEXT,
                line      INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def add_node(self, id: str, file_path: str, name: str, kind: str,
                 language: str, line_start: int = 0, line_end: int = 0):
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(id, file_path, name, kind, language, line_start, line_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (id, file_path, name, kind, language, line_start, line_end),
        )
        self._conn.commit()

    def get_node(self, id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_nodes(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str, kind: str,
                 weight: float = 1.0, file_path: str = "", line: int = 0):
        self._conn.execute(
            "INSERT INTO edges (source, target, kind, weight, file_path, line) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source, target, kind, weight, file_path, line),
        )
        self._conn.commit()

    def get_edges_from(self, source: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source = ?", (source,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edges_to(self, target: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE target = ?", (target,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_edges(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM edges").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        node_count = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM nodes"
        ).fetchone()["cnt"]
        edge_count = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM edges"
        ).fetchone()["cnt"]
        return {"node_count": node_count, "edge_count": edge_count}

    def clear(self):
        self._conn.executescript(
            "DELETE FROM nodes; DELETE FROM edges; DELETE FROM meta;"
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
