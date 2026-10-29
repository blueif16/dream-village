"""gemini-embedding-2 embeddings (docs/memora-plan-addendum.md §5, docs/port-plan.md 1b).

embed(texts, kind) -> one 3072-dim vector per text, in input order.
- `kind` selects the task prefix (the model has no task_type): consolidation / document / query.
- One `types.Content` per text: a bare list of strings is merged by the SDK into ONE embedding.
  At most 100 texts per request, and the returned count is checked against the input count.
- Cached by sha256 of the prefixed text in <root>/embed_cache.sqlite (thread-safe, shared across runs).
- Each API call appends a ledger line (stage="embed") to <root>/ledger.jsonl. 429 / 5xx / connection
  errors are retried with exponential backoff + jitter, and each retry is logged to <root>/logs/retries.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Kind = Literal["consolidation", "document", "query"]

MODEL = "gemini-embedding-2"
DIMS = 3072  # model default; kept (addendum §5)
MAX_PER_REQUEST = 100
EMBED_USD_PER_MTOK = 0.20  # UNVERIFIED (third-party listing, addendum §5): confirm at ai.google.dev/gemini-api/docs/pricing
CHARS_PER_TOKEN = 4  # token estimate for the ledger

# addendum §5 prefix scheme; each prefix is followed directly by the text
PREFIXES: dict[str, str] = {
    "consolidation": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
    "query": "task: question answering | query: ",
}

_file_lock = threading.Lock()


class EmbeddingCountError(RuntimeError):
    """The API returned a different number of vectors than texts sent (or a vector of the wrong size)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _retryable(exc: Exception) -> bool:
    import httpx
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, httpx.TransportError)


class Embedder:
    def __init__(self, root="runs", *, run_id: str = "dev", client=None, max_attempts: int = 6, sleep=time.sleep):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.max_attempts = max_attempts
        self._client = client
        self._sleep = sleep
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.root / "embed_cache.sqlite"), check_same_thread=False)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, model TEXT, vec BLOB)")
            self._db.commit()

    # ------------------------------------------------------------------ public

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        if kind not in PREFIXES:
            raise ValueError(f"unknown embedding kind {kind!r}; expected one of {sorted(PREFIXES)}")
        prefixed = [PREFIXES[kind] + t for t in texts]
        if not prefixed:
            return []
        keys = [_sha256(p) for p in prefixed]
        found = self._cache_get(keys)
        missing = list(dict.fromkeys(p for p, k in zip(prefixed, keys) if k not in found))
        for i in range(0, len(missing), MAX_PER_REQUEST):
            chunk = missing[i : i + MAX_PER_REQUEST]
            vectors = self._request(chunk)
            fresh = {_sha256(p): v for p, v in zip(chunk, vectors)}
            self._cache_put(fresh)
            found.update(fresh)
        return [found[k] for k in keys]

    # ------------------------------------------------------------------ API

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                from google import genai
                from google.genai import types

                # finite read timeout (ms): an unbounded read once parked threads in SSL_read for an hour
                self._client = genai.Client(
                    api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=60_000)
                )
            return self._client

    def _request(self, chunk: list[str]) -> list[list[float]]:
        from google.genai import types

        contents = [types.Content(parts=[types.Part(text=t)]) for t in chunk]
        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            try:
                response = self.client.models.embed_content(model=MODEL, contents=contents)
                break
            except Exception as exc:
                if not _retryable(exc) or attempt >= self.max_attempts:
                    raise
                delay = min(60.0, 2.0 ** (attempt - 1)) * (1.0 + 0.25 * random.random())
                _append_jsonl(
                    self.root / "logs" / "retries.jsonl",
                    {"ts": _now(), "run_id": self.run_id, "stage": "embed", "attempt": attempt,
                     "error": f"{type(exc).__name__}: {exc}"[:500], "sleep_s": delay},
                )
                self._sleep(delay)
        latency_ms = (time.monotonic() - started) * 1000.0

        vectors = [list(e.values or []) for e in (response.embeddings or [])]
        if len(vectors) != len(chunk):
            raise EmbeddingCountError(f"{MODEL} returned {len(vectors)} embeddings for {len(chunk)} inputs")
        wrong = sorted({len(v) for v in vectors if len(v) != DIMS})
        if wrong:
            raise EmbeddingCountError(f"{MODEL} returned vectors of size {wrong}, expected {DIMS}")

        chars = sum(len(t) for t in chunk)
        tokens = chars / CHARS_PER_TOKEN
        _append_jsonl(
            self.root / "ledger.jsonl",
            {"run_id": self.run_id, "stage": "embed", "key": _sha256("\n".join(chunk)),
             "cost_usd": tokens * EMBED_USD_PER_MTOK / 1e6, "mode": "sync", "ts": _now(),
             "model": MODEL, "n_texts": len(chunk), "chars": chars, "est_tokens": tokens,
             "latency_ms": latency_ms},
        )
        return vectors

    # ------------------------------------------------------------------ cache

    def _cache_get(self, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(keys))
        with self._lock:
            for i in range(0, len(unique), 500):
                part = unique[i : i + 500]
                rows = self._db.execute(
                    f"SELECT key, vec FROM embeddings WHERE key IN ({','.join('?' * len(part))})", part
                ).fetchall()
                for key, blob in rows:
                    vec = array("d")
                    vec.frombytes(blob)
                    out[key] = vec.tolist()
        return out

    def _cache_put(self, vectors: dict[str, list[float]]) -> None:
        rows = [(k, MODEL, array("d", v).tobytes()) for k, v in vectors.items()]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO embeddings (key, model, vec) VALUES (?, ?, ?)", rows)
            self._db.commit()


# ---------------------------------------------------------------------- process-wide default

_default: Embedder | None = None
_default_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _default
    with _default_lock:
        if _default is None:
            from memobench.runtime import REPO  # also loads .env (GEMINI_API_KEY)

            _default = Embedder(
                root=os.environ.get("MEMOBENCH_ROOT", str(REPO / "runs")),
                run_id=os.environ.get("MEMOBENCH_RUN_ID", "dev"),
                # Gemini per-minute quotas answer 429 for a while; wait them out rather than fail the caller
                max_attempts=int(os.environ.get("MEMOBENCH_EMBED_MAX_ATTEMPTS", "6")),
            )
        return _default


def embed(texts: list[str], kind: Kind) -> list[list[float]]:
    return get_embedder().embed(list(texts), kind)
