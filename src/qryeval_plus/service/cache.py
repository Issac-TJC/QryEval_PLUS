"""Version-aware SQLite cache for complete service responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict


class ResponseCache:
    def __init__(self, path: str):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            self._db.commit()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("Unable to open response cache '{}': {}".format(self.path, exc)) from exc

    @staticmethod
    def key(*, corpus_version: str, config_hash: str, policy: str, question: str) -> str:
        payload = {
            "corpus_version": corpus_version,
            "config_hash": config_hash,
            "policy": policy,
            "question": " ".join(question.split()),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def get(self, key: str) -> Dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT response_json FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Cached service response is corrupt for key {}.".format(key[:12])) from exc
        if not isinstance(value, dict):
            raise RuntimeError("Cached service response is not an object.")
        return value

    def put(self, key: str, response: Dict[str, Any]) -> None:
        payload = json.dumps(response, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses(cache_key, response_json) VALUES (?, ?)",
                (key, payload),
            )
            self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()
