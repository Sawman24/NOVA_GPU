"""Persistent storage for the Gateway: sessions, messages, long-term memory, cloud usage.

SQLite for Phase 1 (zero setup, one file in data/). The public methods are the
contract; a PostgreSQL + pgvector backend can replace this class later without
changing the API.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    channel     TEXT NOT NULL DEFAULT 'web',
    status      TEXT NOT NULL DEFAULT 'active',
    summary     TEXT NOT NULL DEFAULT '',
    checkpoint  TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL DEFAULT 'fact',
    text            TEXT NOT NULL,
    importance      REAL NOT NULL DEFAULT 0.5,
    source_session  TEXT,
    embedding       BLOB,
    dim             INTEGER NOT NULL DEFAULT 0,
    use_count       INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_used_at    REAL,
    deleted         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    purpose         TEXT NOT NULL DEFAULT '',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    ok              INTEGER NOT NULL DEFAULT 1,
    error           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL
);
"""

MERGE_THRESHOLD = 0.92  # cosine similarity above which a new memory updates an old one


def _now() -> float:
    return time.time()


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("checkpoint", "meta"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                d[k] = {}
    d.pop("embedding", None)
    return d


class Store:
    def __init__(self, path: str):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self._cache: dict[str, Any] | None = None  # vector index cache

    def snapshot(self, dest: str) -> None:
        """A consistent copy of the whole database (memory, chats, voices) while it stays in use."""
        with self._lock:
            out = sqlite3.connect(dest)
            try:
                self.db.backup(out)
            finally:
                out.close()

    def _migrate(self) -> None:
        """People: who a memory/chat belongs to. '' = the household's owner (everything made before people existed).
        scope: 'private' (that person, plus the admin) or 'shared' (everyone NOVA knows, not guests)."""
        def cols(table: str) -> set[str]:
            return {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
        if "person" not in cols("memories"):
            self.db.execute("ALTER TABLE memories ADD COLUMN person TEXT NOT NULL DEFAULT ''")
        if "scope" not in cols("memories"):
            self.db.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'private'")
        if "person" not in cols("sessions"):
            self.db.execute("ALTER TABLE sessions ADD COLUMN person TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _visible(person: str, scope: str, viewer: str | None, admin: bool) -> bool:
        """viewer None = no filter (internal/admin tools). Guests (viewer '') see nothing."""
        if viewer is None or admin:
            return True
        if not viewer:
            return False
        return person == viewer or scope == "shared"

    def _exec(self, sql: str, args: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.db.execute(sql, tuple(args))
            self.db.commit()
            return cur

    def _all(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, tuple(args)).fetchall()

    def _one(self, sql: str, args: Iterable = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.db.execute(sql, tuple(args)).fetchone()

    # ---------------------------------------------------------------- sessions
    def create_session(self, title: str = "New session", channel: str = "web", person: str = "") -> dict:
        sid = uuid.uuid4().hex[:12]
        t = _now()
        self._exec(
            "INSERT INTO sessions(id,title,channel,person,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (sid, title or "New session", channel or "web", person or "", t, t),
        )
        return self.get_session(sid)

    def get_session(self, sid: str) -> dict | None:
        return _row(self._one("SELECT * FROM sessions WHERE id=?", (sid,)))

    def list_sessions(self, limit: int = 50, status: str | None = None, person: str | None = None) -> list[dict]:
        """person: only that person's chats (None = everyone's)."""
        where, args = [], []
        if status:
            where.append("status=?")
            args.append(status)
        if person is not None:
            where.append("person=?")
            args.append(person)
        rows = self._all(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS message_count "
            f"FROM sessions s {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY updated_at DESC LIMIT ?",
            (*args, limit),
        )
        return [_row(r) for r in rows]

    def update_session(self, sid: str, **fields: Any) -> dict | None:
        allowed = {"title", "status", "summary", "checkpoint", "channel", "person"}
        sets, args = [], []
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k == "checkpoint":
                v = json.dumps(v)
            sets.append(f"{k}=?")
            args.append(v)
        if sets:
            sets.append("updated_at=?")
            args.append(_now())
            self._exec(f"UPDATE sessions SET {', '.join(sets)} WHERE id=?", (*args, sid))
        return self.get_session(sid)

    def delete_session(self, sid: str) -> bool:
        """Remove a conversation and its messages. Memories learned from it are kept."""
        with self._lock:
            self.db.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            cur = self.db.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self.db.commit()
            return cur.rowcount > 0

    def add_message(self, sid: str, role: str, content: str, meta: dict | None = None) -> dict:
        t = _now()
        cur = self._exec(
            "INSERT INTO messages(session_id,role,content,meta,created_at) VALUES(?,?,?,?,?)",
            (sid, role, content, json.dumps(meta or {}), t),
        )
        self._exec("UPDATE sessions SET updated_at=? WHERE id=?", (t, sid))
        return _row(self._one("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)))

    def get_messages(self, sid: str, limit: int | None = None, after_id: int = 0) -> list[dict]:
        if limit:
            rows = self._all(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id DESC LIMIT ?) "
                "ORDER BY id ASC",
                (sid, after_id, limit),
            )
        else:
            rows = self._all(
                "SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id ASC", (sid, after_id)
            )
        return [_row(r) for r in rows]

    def count_messages(self, sid: str, after_id: int = 0) -> int:
        r = self._one("SELECT COUNT(*) AS n FROM messages WHERE session_id=? AND id>?", (sid, after_id))
        return int(r["n"]) if r else 0

    # ---------------------------------------------------------------- memory
    def _load_index(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        rows = self._all(
            "SELECT id, kind, importance, created_at, dim, embedding, person, scope FROM memories "
            "WHERE deleted=0 AND embedding IS NOT NULL"
        )
        by_dim: dict[int, dict[str, Any]] = {}
        for r in rows:
            d = by_dim.setdefault(r["dim"], {"ids": [], "kinds": [], "imp": [], "created": [], "vecs": [],
                                             "person": [], "scope": []})
            d["ids"].append(r["id"])
            d["person"].append(r["person"] or "")
            d["scope"].append(r["scope"] or "private")
            d["kinds"].append(r["kind"])
            d["imp"].append(r["importance"])
            d["created"].append(r["created_at"])
            d["vecs"].append(np.frombuffer(r["embedding"], dtype=np.float32))
        for d in by_dim.values():
            m = np.vstack(d["vecs"])
            d["mat"] = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
            d["imp"] = np.array(d["imp"], dtype=np.float32)
            d["created"] = np.array(d["created"], dtype=np.float64)
            del d["vecs"]
        self._cache = by_dim
        return by_dim

    def _invalidate(self) -> None:
        self._cache = None

    @staticmethod
    def _norm(vec: list[float]) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def add_memory(
        self,
        text: str,
        embedding: list[float],
        kind: str = "fact",
        importance: float = 0.5,
        source_session: str | None = None,
        person: str = "",
        scope: str = "private",
    ) -> dict:
        """Store a memory. Near-duplicates (same person) update the existing memory instead (newest wording wins)."""
        q = self._norm(embedding)
        idx = self._load_index().get(len(q))
        importance = float(min(max(importance, 0.0), 1.0))
        scope = "shared" if scope == "shared" else "private"
        if idx is not None and len(idx["ids"]):
            sims = (idx["mat"] @ q).copy()
            sims[[p != (person or "") for p in idx["person"]]] = -1.0
            best = int(np.argmax(sims))
            if float(sims[best]) >= MERGE_THRESHOLD:
                mid = idx["ids"][best]
                old = self._one("SELECT importance FROM memories WHERE id=?", (mid,))
                self._exec(
                    "UPDATE memories SET text=?, importance=?, embedding=?, updated_at=? WHERE id=?",
                    (text, max(importance, old["importance"] if old else 0), q.tobytes(), _now(), mid),
                )
                self._invalidate()
                m = self.get_memory(mid)
                m["merged"] = True
                return m
        mid = uuid.uuid4().hex[:12]
        t = _now()
        self._exec(
            "INSERT INTO memories(id,kind,text,importance,source_session,embedding,dim,created_at,updated_at,person,scope) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (mid, kind, text, importance, source_session, q.tobytes(), len(q), t, t, person or "", scope),
        )
        self._invalidate()
        m = self.get_memory(mid)
        m["merged"] = False
        return m

    def get_memory(self, mid: str) -> dict | None:
        return _row(self._one("SELECT * FROM memories WHERE id=?", (mid,)))

    def search_memories(
        self,
        embedding: list[float],
        k: int = 6,
        min_score: float = 0.0,
        kinds: list[str] | None = None,
        touch: bool = True,
        viewer: str | None = None,
        admin: bool = False,
    ) -> list[dict]:
        """Rank by meaning (80%), importance (10%) and recency (10%). viewer/admin: whose memories may be seen."""
        q = self._norm(embedding)
        idx = self._load_index().get(len(q))
        if idx is None or not len(idx["ids"]):
            return []
        sims = idx["mat"] @ q
        age_days = (_now() - idx["created"]) / 86400.0
        recency = np.exp(-age_days / 30.0)
        score = 0.8 * sims + 0.1 * idx["imp"] + 0.1 * recency
        order = np.argsort(-score)
        out: list[dict] = []
        for i in order:
            if float(sims[i]) < min_score:
                continue
            if kinds and idx["kinds"][i] not in kinds:
                continue
            if not self._visible(idx["person"][i], idx["scope"][i], viewer, admin):
                continue
            m = self.get_memory(idx["ids"][i])
            if m is None:
                continue
            m["similarity"] = round(float(sims[i]), 4)
            m["score"] = round(float(score[i]), 4)
            out.append(m)
            if len(out) >= k:
                break
        if touch and out:
            t = _now()
            with self._lock:
                self.db.executemany(
                    "UPDATE memories SET use_count=use_count+1, last_used_at=? WHERE id=?",
                    [(t, m["id"]) for m in out],
                )
                self.db.commit()
        return out

    def list_memories(self, limit: int = 100, offset: int = 0, kind: str | None = None,
                      viewer: str | None = None, admin: bool = False) -> list[dict]:
        where, args = ["deleted=0"], []
        if kind:
            where.append("kind=?")
            args.append(kind)
        if viewer is not None and not admin:
            if not viewer:
                return []
            where.append("(person=? OR scope='shared')")
            args.append(viewer)
        rows = self._all(f"SELECT * FROM memories WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                         (*args, limit, offset))
        return [_row(r) for r in rows]

    def count_memories(self) -> int:
        r = self._one("SELECT COUNT(*) AS n FROM memories WHERE deleted=0")
        return int(r["n"]) if r else 0

    def update_memory(
        self,
        mid: str,
        text: str | None = None,
        importance: float | None = None,
        kind: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict | None:
        sets, args = [], []
        if text is not None:
            sets.append("text=?")
            args.append(text)
        if importance is not None:
            sets.append("importance=?")
            args.append(float(min(max(importance, 0.0), 1.0)))
        if kind is not None:
            sets.append("kind=?")
            args.append(kind)
        if embedding is not None:
            q = self._norm(embedding)
            sets += ["embedding=?", "dim=?"]
            args += [q.tobytes(), len(q)]
        if sets:
            sets.append("updated_at=?")
            args.append(_now())
            self._exec(f"UPDATE memories SET {', '.join(sets)} WHERE id=?", (*args, mid))
            self._invalidate()
        return self.get_memory(mid)

    def delete_memory(self, mid: str) -> bool:
        cur = self._exec("UPDATE memories SET deleted=1, updated_at=? WHERE id=? AND deleted=0", (_now(), mid))
        self._invalidate()
        return cur.rowcount > 0

    # ---------------------------------------------------------------- cloud usage
    def log_escalation(
        self,
        provider: str,
        model: str,
        purpose: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        ok: bool = True,
        error: str = "",
    ) -> None:
        self._exec(
            "INSERT INTO escalations(provider,model,purpose,input_tokens,output_tokens,ok,error,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (provider, model, purpose, int(input_tokens), int(output_tokens), 1 if ok else 0, error[:500], _now()),
        )

    def escalations_today(self, provider: str) -> int:
        lt = time.localtime()
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        r = self._one(
            "SELECT COUNT(*) AS n FROM escalations WHERE provider=? AND created_at>=? AND ok=1",
            (provider, midnight),
        )
        return int(r["n"]) if r else 0

    def usage_summary(self, days: int = 30) -> list[dict]:
        since = _now() - days * 86400
        rows = self._all(
            "SELECT provider, model, COUNT(*) AS calls, SUM(ok) AS ok_calls, "
            "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens "
            "FROM escalations WHERE created_at>=? GROUP BY provider, model ORDER BY calls DESC",
            (since,),
        )
        return [dict(r) for r in rows]


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
