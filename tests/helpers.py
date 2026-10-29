"""Shared test helpers: request builder, JSONL reader, the default fake reply's hand-computed cost."""

from __future__ import annotations

import json
from pathlib import Path

from memobench.llm import LLM, Req, Result

MODEL = "claude-haiku-5-5"

# The fake's default reply uses 1000 input + 200 output tokens, no cache tokens (tests/fakes.usage_payload).
# Short tier ($/MTok): input 0.10, output 0.50.  Batch: x 0.5.
SYNC_COST = 1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6
BATCH_COST = (1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6) * 0.5


def make_req(prompt: str, *, meta: dict | None = None, run_id: str = "r1", **params) -> Req:
    """A complete Messages API request whose last user message is `prompt`."""
    p = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}
    p.update(params)
    m = {"run_id": run_id, "system": "memora", "conv_id": "conv-1", "question_id": f"qid-{prompt[:20]}", "stage": "answer"}
    m.update(meta or {})
    return Req(params=p, meta=m)


def run_batch(llm: LLM, reqs: list[Req], **kw) -> dict[str, Result]:
    """run(mode="batch") with an estimate that never trips G3 (approval) or S3 (overrun)."""
    kw.setdefault("estimate_usd", 1.0)
    return llm.run(reqs, mode="batch", **kw)


def run_sync(llm: LLM, reqs: list[Req], **kw) -> dict[str, Result]:
    kw.setdefault("estimate_usd", 1.0)
    return llm.run(reqs, mode="sync", **kw)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
