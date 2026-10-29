"""LLM layer: guards, response cache, real-time + Message Batches execution, cost ledger, request log.

Spec: docs/llm-layer-spec.md (clause ids G/E/C/S/B/T/L are cited inline). One instance is safe to share across
threads (T1): sqlite access, JSONL appends, and spend counters are each serialized by a lock.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, TypeVar

import anthropic
import httpx
from google.genai import errors as genai_errors

from memobench.pricing import GEMINI_RATES, cost_usd

T = TypeVar("T")

FROZEN_PARAMS = ("temperature", "top_p", "top_k", "seed")  # G1: 400 on Haiku 5.5
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
OVERRUN_FACTOR = 1.5  # S3
BACKOFF_BASE_S, BACKOFF_MAX_S = 1.0, 60.0  # S2
EST_CHARS_PER_TOKEN, EST_MAX_OUTPUT = 3, 1024  # E1
API_ERRORS = (anthropic.APIError, genai_errors.APIError, httpx.TransportError)
GEMINI_PARAMS = ("model", "max_tokens", "messages", "system", "output_config")  # all the Gemini path translates
# finish reason -> Anthropic stop_reason; anything else (SAFETY, RECITATION, OTHER, no candidate) -> "refusal"
GEMINI_STOP = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "CONTINUATION": "max_tokens"}
GEMINI_TIMEOUT_MS = 600_000


class BudgetExceeded(Exception):
    pass


class NeedsApproval(Exception):
    pass


class CostOverrun(Exception):
    pass


@dataclass
class Req:
    params: dict
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """C2: sha256 of the canonical params JSON; meta is not part of the key."""
        canonical = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Result:
    key: str
    text: str
    message: dict
    stop_reason: str | None
    usage: dict
    cost_usd: float
    latency_ms: float
    cached: bool
    mode: Literal["sync", "batch"]
    batch_id: str | None
    error: str | None


@dataclass
class _Run:
    """Spend of one run() call, for the S3 overrun stop."""

    limit_usd: float
    spent_usd: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _visible_text(message: dict) -> str:
    """L2: text blocks only (thinking blocks are skipped)."""
    return "".join(b.get("text", "") for b in message.get("content") or [] if b.get("type") == "text")


def _failed(key: str, mode: Literal["sync", "batch"], batch_id: str | None, error: str) -> Result:
    return Result(key, "", {}, None, {}, 0.0, 0.0, False, mode, batch_id, error)


def _retryable(e: BaseException) -> bool:
    """S2: 429, any 5xx (incl. 529 overloaded), connection errors and timeouts (Anthropic or Gemini)."""
    if isinstance(e, (anthropic.APIConnectionError, httpx.TransportError)):
        return True
    if isinstance(e, genai_errors.APIError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, anthropic.APIStatusError) and (e.status_code == 429 or e.status_code >= 500)


def _retry_after(e: BaseException) -> float | None:
    response = getattr(e, "response", None)
    if response is None:
        return None
    for header, seconds in (("retry-after-ms", 1e-3), ("retry-after", 1.0)):
        try:
            return max(0.0, float(response.headers[header]) * seconds)
        except (KeyError, ValueError):
            pass
    return None


def _describe(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _is_gemini(params: dict) -> bool:
    return str(params.get("model", "")).startswith("gemini-")


def _joined_text(content: str | list[dict]) -> str:
    """Anthropic content (a string or text blocks) as one string; cache_control is Anthropic-only and dropped."""
    if isinstance(content, str):
        return content
    if any(b.get("type") != "text" for b in content):
        raise ValueError(f"only text blocks have a Gemini translation, got {[b.get('type') for b in content]}")
    return "".join(b["text"] for b in content)


def _gemini_request(params: dict) -> dict:
    """G5: Anthropic-shaped params -> generate_content kwargs carrying the same prompt text. Raises on anything
    without a translation, so nothing is silently dropped."""
    from google.genai import types

    extra = sorted(set(params) - set(GEMINI_PARAMS))
    output_config = dict(params.get("output_config") or {})
    effort, fmt = output_config.pop("effort", None), output_config.pop("format", None)
    if extra or output_config:
        raise ValueError(f"params {extra + sorted(output_config)} have no Gemini translation")
    config: dict[str, Any] = {"max_output_tokens": params["max_tokens"]}
    if params.get("system"):
        config["system_instruction"] = _joined_text(params["system"])
    if effort:
        config["thinking_config"] = types.ThinkingConfig(thinking_level=effort.upper())
    if fmt:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = fmt["schema"]
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": _joined_text(m["content"])}]}
        for m in params["messages"]
    ]
    return {"model": params["model"], "contents": contents, "config": types.GenerateContentConfig(**config)}


def _gemini_message(resp: Any) -> dict:
    """A generate_content response in the Anthropic message shape the cache, request log and Result read."""
    cand = resp.candidates[0] if resp.candidates else None
    parts = (cand.content.parts or []) if cand is not None and cand.content is not None else []
    finish = cand.finish_reason.name if cand is not None and cand.finish_reason is not None else None
    u = resp.usage_metadata
    prompt, cached = ((u.prompt_token_count or 0), (u.cached_content_token_count or 0)) if u else (0, 0)
    output = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
    return {
        "id": resp.response_id,
        "type": "message",
        "role": "assistant",
        "model": resp.model_version,
        "content": [{"type": "text", "text": "".join(p.text for p in parts if p.text and not p.thought)}],
        "stop_reason": GEMINI_STOP.get(finish, "refusal"),
        "gemini_finish_reason": finish,
        "usage": {
            "input_tokens": prompt - cached,
            "output_tokens": output,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached,
        },
    }


class LLM:
    def __init__(
        self,
        root: str = "runs",
        *,
        run_id: str,
        budget_usd: float = 10.0,
        approval_usd: float = 50.0,
        client: Any = None,
        gemini_client: Any = None,
        allowed_models: tuple[str, ...] = ("claude-haiku-5-5",),
        max_batch_requests: int = 100_000,
        max_batch_bytes: int = 200_000_000,
        max_attempts: int = 6,
        gemini_concurrency: int = 64,
        sleep=time.sleep,
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.budget_usd = budget_usd
        self.approval_usd = approval_usd
        self.client = client if client is not None else anthropic.Anthropic(max_retries=0)
        self._gemini = gemini_client
        self.allowed_models = tuple(allowed_models)
        self.max_batch_requests = max_batch_requests
        self.max_batch_bytes = max_batch_bytes
        self.max_attempts = max_attempts
        self.gemini_concurrency = gemini_concurrency
        self.sleep = sleep

        self._batches_dir = self.root / "batches"
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self._batches_dir.mkdir(parents=True, exist_ok=True)
        self._requests_log = self.root / "logs" / "requests.jsonl"
        self._retries_log = self.root / "logs" / "retries.jsonl"
        self._ledger = self.root / "ledger.jsonl"

        self._lock = threading.Lock()  # spend counters + ledger tail position
        self._io_lock = threading.Lock()  # JSONL appends
        self._db_lock = threading.Lock()  # the shared sqlite connection
        self._spent = 0.0
        self._ledger_pos, self._ledger_sum = 0, 0.0

        self._db = sqlite3.connect(self.root / "cache.sqlite", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, message TEXT NOT NULL, mode TEXT NOT NULL, batch_id TEXT, ts TEXT NOT NULL)"
            )

    # ------------------------------------------------------------------ public API

    def call(self, req: Req) -> Result:
        """One real-time request, cache-first (C1). API errors are raised after S2 retries."""
        self._guard([req])
        hit = self._cached(req.key)
        return hit if hit is not None else self._fetch(req, run=None)

    def lookup(self, req: Req) -> Result | None:
        """C6: cache read only. The cached Result (as call() returns it on a hit), or None; never calls the API."""
        return self._cached(req.key)

    def run(
        self,
        reqs: Iterable[Req],
        mode: Literal["sync", "batch"],
        *,
        estimate_usd: float | None = None,
        approved: bool = False,
        concurrency: int = 16,
        poll_s: float = 30,
        batch_timeout_s: float | None = None,
    ) -> dict[str, Result]:
        """Run every req (cache-first, duplicates once) and return {req.key: Result}.

        Per-request API failures come back as Result.error; guard violations raise
        (ValueError, NeedsApproval, BudgetExceeded, CostOverrun)."""
        if mode not in ("sync", "batch"):
            raise ValueError(f"mode must be 'sync' or 'batch', not {mode!r}")
        reqs = list(reqs)
        self._guard(reqs)  # G1/G2 on every req before anything is sent
        unique: dict[str, Req] = {}
        for r in reqs:
            unique.setdefault(r.key, r)  # C3
        out: dict[str, Result] = {}
        misses: list[Req] = []
        for key, r in unique.items():
            hit = self._cached(key)
            if hit is not None:
                out[key] = hit
            else:
                misses.append(r)

        est = self._estimate(misses, batch=mode == "batch") if estimate_usd is None else estimate_usd
        print(f"[memobench] {self.run_id} {mode} n={len(misses)}/{len(reqs)} est=${est:.4f}", flush=True)  # E1
        if est > self.approval_usd and not approved:  # G3
            raise NeedsApproval(f"estimate ${est:.4f} > approval_usd ${self.approval_usd:.2f}; pass approved=True")

        run = _Run(limit_usd=OVERRUN_FACTOR * est)
        if mode == "batch" and any(_is_gemini(r.params) for r in misses):  # G6: no Gemini batches; real-time instead
            out.update(self._run_sync([r for r in misses if _is_gemini(r.params)], run, max(concurrency, self.gemini_concurrency)))
            misses = [r for r in misses if not _is_gemini(r.params)]
        if misses and mode == "sync":
            out.update(self._run_sync(misses, run, concurrency))
        elif misses:
            out.update(self._run_batch(misses, run, concurrency, poll_s, batch_timeout_s))
        self._check_overrun(run)
        return out

    def estimate(self, reqs: Iterable[Req], *, batch: bool) -> float:
        """E1: rough USD for the cache misses among `reqs` (duplicates counted once). No API calls."""
        unique = {r.key: r for r in reqs}
        return self._estimate([r for key, r in unique.items() if self._cached(key) is None], batch=batch)

    @property
    def spent_usd(self) -> float:
        """L3: this instance's paid spend (cache hits excluded)."""
        with self._lock:
            return self._spent

    @property
    def total_usd(self) -> float:
        """L3: sum of root/ledger.jsonl (all runs, any process). Reads only lines appended since the last call."""
        with self._lock:
            if self._ledger.exists():
                with open(self._ledger, "rb") as f:
                    f.seek(self._ledger_pos)
                    tail = f.read()
                complete = tail[: tail.rfind(b"\n") + 1]  # never consume a half-written last line
                self._ledger_sum += sum(json.loads(line)["cost_usd"] for line in complete.splitlines() if line.strip())
                self._ledger_pos += len(complete)
            return self._ledger_sum

    # ------------------------------------------------------------------ guards + estimate

    def _guard(self, reqs: list[Req]) -> None:
        for r in reqs:
            frozen = [p for p in FROZEN_PARAMS if p in r.params]
            if frozen:
                raise ValueError(f"frozen sampling params {frozen} are not allowed (meta={r.meta})")  # G1
            if r.params.get("model") not in self.allowed_models:
                raise ValueError(f"model {r.params.get('model')!r} not in allowed_models {self.allowed_models}")  # G2
            if _is_gemini(r.params):
                if r.params["model"] not in GEMINI_RATES:
                    raise ValueError(f"no pricing for {r.params['model']!r} in memobench.pricing.GEMINI_RATES")
                _gemini_request(r.params)  # G5: raises before anything is sent

    def _estimate(self, reqs: list[Req], *, batch: bool) -> float:
        total = 0.0
        for r in reqs:
            prompt_chars = len(json.dumps([r.params.get("system"), r.params["messages"]]))
            usage = {
                "input_tokens": math.ceil(prompt_chars / EST_CHARS_PER_TOKEN),
                "output_tokens": min(r.params["max_tokens"], EST_MAX_OUTPUT),
            }
            total += cost_usd(usage, batch=batch, model=r.params["model"])
        return total

    def _check_overrun(self, run: _Run | None) -> None:
        if run is None:
            return
        with self._lock:
            spent, limit = run.spent_usd, run.limit_usd
        if spent > limit:  # S3
            raise CostOverrun(f"run spend ${spent:.4f} > {OVERRUN_FACTOR} x estimate (${limit:.4f})")

    def _check_dispatch(self, run: _Run | None) -> None:
        """Before every request / batch is sent: S3 overrun stop (inside run()), then G4 budget."""
        self._check_overrun(run)
        total = self.total_usd
        if total >= self.budget_usd:
            raise BudgetExceeded(f"total ${total:.4f} >= budget ${self.budget_usd:.4f}")

    # ------------------------------------------------------------------ cache, log, ledger

    def _cached(self, key: str) -> Result | None:
        """C1/C5: a hit costs nothing and writes no log or ledger line."""
        t0 = time.perf_counter()
        with self._db_lock:
            row = self._db.execute("SELECT message, mode, batch_id FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        message = json.loads(row[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        return Result(
            key, _visible_text(message), message, message.get("stop_reason"), message.get("usage") or {},
            0.0, latency_ms, True, row[1], row[2], None,
        )  # fmt: skip

    def _append(self, path: Path, record: dict) -> None:
        line = json.dumps(record) + "\n"
        with self._io_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def _record(
        self,
        key: str,
        params: dict | None,
        meta: dict,
        message: dict,
        mode: Literal["sync", "batch"],
        batch_id: str | None,
        latency_ms: float,
        run: _Run | None,
    ) -> Result:
        """A paid response: ledger (L3) -> cache (C1, success only) -> request log (L1). Returns its Result."""
        usage = message.get("usage") or {}
        cost = cost_usd(usage, batch=mode == "batch", model=(params or {}).get("model") or message.get("model"))
        text = _visible_text(message)
        ts = _now()
        self._append(
            self._ledger,
            {"run_id": self.run_id, "stage": meta.get("stage"), "key": key, "cost_usd": cost, "mode": mode, "ts": ts},
        )
        with self._lock:
            self._spent += cost
            if run is not None:
                run.spent_usd += cost
        with self._db_lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)", (key, json.dumps(message), mode, batch_id, ts)
            )
        self._append(
            self._requests_log,
            {
                "run_id": self.run_id,
                **{k: meta.get(k) for k in ("system", "conv_id", "question_id", "stage")},
                "model": params.get("model") if params else message.get("model"),
                "effort": ((params or {}).get("output_config") or {}).get("effort"),
                "mode": mode,
                "batch_id": batch_id,
                "key": key,
                "request": params,
                "response": message,
                "usage": {k: usage.get(k) or 0 for k in USAGE_FIELDS},
                "visible_text_chars": len(text),
                "stop_reason": message.get("stop_reason"),  # L4: recorded, never raised
                "latency_ms": latency_ms,
                "cost_usd": cost,
                "ts": ts,
            },
        )
        return Result(key, text, message, message.get("stop_reason"), usage, cost, latency_ms, False, mode, batch_id, None)

    # ------------------------------------------------------------------ real-time path

    def _with_retries(self, fn: Callable[[], T], key: str) -> T:
        """S2: retry 429/5xx/connection errors (backoff + jitter, or `retry-after`), max_attempts calls in total."""
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except API_ERRORS as e:
                if attempt >= self.max_attempts or not _retryable(e):
                    raise
                wait = _retry_after(e)
                if wait is None:
                    wait = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
                self._append(
                    self._retries_log,
                    {"ts": _now(), "key": key, "attempt": attempt, "error": _describe(e), "sleep_s": wait},
                )
                self.sleep(wait)
        raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")

    def _fetch(self, req: Req, run: _Run | None) -> Result:
        self._check_dispatch(run)

        def attempt() -> tuple[dict, float]:
            t0 = time.perf_counter()
            if _is_gemini(req.params):
                msg = _gemini_message(self._gemini_client().models.generate_content(**_gemini_request(req.params)))
            else:
                msg = self.client.messages.create(**req.params).to_dict(mode="json")
            return msg, (time.perf_counter() - t0) * 1000

        msg, latency_ms = self._with_retries(attempt, req.key)
        return self._record(req.key, req.params, req.meta, msg, "sync", None, latency_ms, run)

    def _gemini_client(self) -> Any:
        with self._lock:
            if self._gemini is None:
                from google import genai
                from google.genai import types

                self._gemini = genai.Client(
                    api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
                )
            return self._gemini

    def _run_sync(self, reqs: list[Req], run: _Run, concurrency: int) -> dict[str, Result]:
        """S1: `concurrency` workers; each checks S3/G4 before sending. API failures become Result.error."""

        def task(r: Req) -> Result:
            try:
                return self._fetch(r, run)
            except API_ERRORS as e:
                return _failed(r.key, "sync", None, _describe(e))

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [(r.key, pool.submit(task, r)) for r in reqs]
        out: dict[str, Result] = {}
        stop: Exception | None = None
        for key, fut in futures:
            try:
                out[key] = fut.result()
            except (BudgetExceeded, CostOverrun) as e:  # finished requests stay cached + logged
                stop = stop or e
        if stop is not None:
            raise stop
        return out

    # ------------------------------------------------------------------ batch path

    def _run_batch(
        self, misses: list[Req], run: _Run, concurrency: int, poll_s: float, timeout_s: float | None
    ) -> dict[str, Result]:
        todo = {r.key: r for r in misses}  # unresolved requests
        out: dict[str, Result] = {}
        submissions = dict.fromkeys(todo, 0)  # batch submissions per key in this run (B6: at most 2)

        # B4: unfinished batches from earlier (possibly crashed) runs that hold some of these keys
        pending = [m for m in self._open_manifests() if todo.keys() & set(m["custom_ids"])]
        for m in pending:
            for key in m["custom_ids"]:
                if key in submissions:
                    submissions[key] += 1
        fresh = [r for r in misses if submissions[r.key] == 0]

        # B8: the first request of each prefix group runs real-time, to completion, before the batch
        warm: dict[Any, Req] = {}
        for r in fresh:
            if r.meta.get("prefix_group") is not None:
                warm.setdefault(r.meta["prefix_group"], r)
        if warm:
            out.update(self._run_sync(list(warm.values()), run, concurrency))
            for r in warm.values():
                del todo[r.key]

        to_submit = [r for r in fresh if r.key in todo]
        to_sync: dict[str, Req] = {}
        while to_submit or pending:
            for r in to_submit:
                submissions[r.key] += 1
            pending += self._submit(to_submit, run)
            retry: dict[str, Req] = {}
            for m, batch, canceled in self._wait(pending, poll_s, timeout_s):
                found = self._collect(m, batch, todo, run)
                for key in m["custom_ids"]:
                    if key not in todo:
                        continue
                    got = found.get(key, ("transient", "missing from batch results"))
                    if isinstance(got, Result):
                        out[key] = got
                    elif got[0] == "invalid" or (not canceled and submissions[key] >= 2):
                        out[key] = _failed(key, "batch", m["batch_id"], got[1])
                    elif canceled:  # B7: timed out -> real-time path
                        to_sync[key] = todo[key]
                        continue
                    else:  # B6: transient -> resubmit once
                        retry[key] = todo[key]
                        continue
                    del todo[key]
            pending, to_submit = [], [r for key, r in retry.items() if key in todo]

        remaining = [r for key, r in to_sync.items() if key in todo]
        if remaining:
            out.update(self._run_sync(remaining, run, concurrency))
        return out

    def _chunks(self, reqs: list[Req]) -> Iterator[list[Req]]:
        """B2: at most max_batch_requests requests and max_batch_bytes of serialized requests per batch."""
        chunk: list[Req] = []
        size = 0
        for r in reqs:
            n = len(json.dumps({"custom_id": r.key, "params": r.params}, separators=(",", ":")).encode("utf-8"))
            if chunk and (len(chunk) >= self.max_batch_requests or size + n > self.max_batch_bytes):
                yield chunk
                chunk, size = [], 0
            chunk.append(r)
            size += n
        if chunk:
            yield chunk

    def _submit(self, reqs: list[Req], run: _Run) -> list[dict]:
        """B1-B3: one batch per chunk (custom_id = req.key); its manifest is on disk before any polling."""
        manifests = []
        for chunk in self._chunks(reqs):
            self._check_dispatch(run)
            batch = self.client.messages.batches.create(
                requests=[{"custom_id": r.key, "params": r.params} for r in chunk]
            )
            m = {
                "batch_id": batch.id,
                "created_at": batch.created_at.isoformat(),
                "run_id": self.run_id,
                "custom_ids": [r.key for r in chunk],
                "metas": {r.key: r.meta for r in chunk},
                "done": False,
            }
            self._write_manifest(m)
            manifests.append(m)
        return manifests

    def _wait(
        self, manifests: list[dict], poll_s: float, timeout_s: float | None
    ) -> Iterator[tuple[dict, Any, bool]]:
        """B5/B7: poll every poll_s; cancel batches still running after timeout_s. Yields (manifest, batch, canceled)
        as each batch reaches "ended"."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        waiting = {m["batch_id"]: m for m in manifests}
        canceled: set[str] = set()
        while waiting:
            for bid, m in list(waiting.items()):
                batch = self._with_retries(lambda: self.client.messages.batches.retrieve(bid), bid)
                if batch.processing_status == "ended":
                    del waiting[bid]
                    yield m, batch, bid in canceled
                elif deadline is not None and bid not in canceled and time.monotonic() >= deadline:
                    self._with_retries(lambda: self.client.messages.batches.cancel(bid), bid)
                    canceled.add(bid)
            if waiting:
                self.sleep(poll_s)

    def _collect(self, m: dict, batch: Any, todo: dict[str, Req], run: _Run) -> dict[str, Result | tuple[str, str]]:
        """B5/B6: match results by custom_id; record every success (it is paid, requested now or not); mark done."""
        bid = m["batch_id"]
        latency_ms = (batch.ended_at - batch.created_at).total_seconds() * 1000 if batch.ended_at else 0.0
        found: dict[str, Result | tuple[str, str]] = {}
        for item in self._with_retries(lambda: self.client.messages.batches.results(bid), bid):
            key, res = item.custom_id, item.result
            if res.type == "succeeded":
                req = todo.get(key)
                found[key] = self._record(
                    key, req.params if req else None, m["metas"].get(key) or {},
                    res.message.to_dict(mode="json"), "batch", bid, latency_ms, run,
                )  # fmt: skip
            elif res.type == "errored":
                kind = "invalid" if res.error.error.type == "invalid_request_error" else "transient"
                found[key] = (kind, f"errored: {res.error.error.type}: {res.error.error.message}")
            else:  # expired / canceled
                found[key] = ("transient", res.type)
        self._write_manifest({**m, "done": True})
        return found

    def _write_manifest(self, m: dict) -> None:
        path = self._batches_dir / f"{m['batch_id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(m))
        tmp.replace(path)  # atomic: a crash never leaves a half-written manifest

    def _open_manifests(self) -> list[dict]:
        manifests = (json.loads(p.read_text()) for p in sorted(self._batches_dir.glob("*.json")))
        return [m for m in manifests if not m.get("done")]
