"""Single-runner durable journal. SQLite guarantees one committed result per content key.

Not a distributed lease queue: callers must not run two runners on the same DB.
The OS lock below enforces that on the tested macOS/Linux platforms.
"""

import fcntl
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def runner_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, sample_id TEXT, "
            "status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT)"
        )
        self.db.commit()

    def enqueue(self, key, sid):
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO jobs(key,sample_id,status) VALUES(?,?,'pending')", (key, sid)
            )

    def recover(self):
        with self.db:
            n = self.db.execute("UPDATE jobs SET status='pending' WHERE status='running'").rowcount
        return n

    def get(self, key):
        row = self.db.execute("SELECT result FROM jobs WHERE key=? AND status='succeeded'", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def start(self, key):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, error=NULL WHERE key=?", (key,)
            )

    def succeed(self, key, result):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status='succeeded',result=?,error=NULL WHERE key=?",
                (json.dumps(result, allow_nan=False), key),
            )

    def fail(self, key, error):
        with self.db:
            self.db.execute("UPDATE jobs SET status='failed',error=? WHERE key=?", (error, key))

    def counts(self):
        return dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())

    def close(self):
        self.db.close()
