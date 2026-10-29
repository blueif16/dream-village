"""Batch-prefetch the answers of a LoCoMo Memora-S run on an EXISTING store (Batch API, 50% off), so the normal
run_memora.py afterwards answers every question from the response cache.

Memora's search answers one question at a time through `LLM.call()`. Capture runs that same search path
(`providers.memora.search.MemoraSearch`) with `memobench.runtime.get_llm` -> CaptureLLM: a cached Result on a hit;
on a miss it records the Req and raises CaptureMiss (never the API; search logs it as a failed answer). The misses
then go out in ONE `LLM.run(mode="batch")`. Only retrieval.strategy=semantic: its retrieval makes no LLM calls,
so the answers are the only requests and capture reproduces them byte-identically.

Usage (from the repo root; overrides = those of the run_memora.py run to prefetch, store already built):
  MEMOBENCH_RUN_ID=<id> uv run python scripts/locomo_answer_prefetch.py memory.memory_store=<store> \
      retrieval.strategy=semantic memory.enable_episodic_memory=True ...
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "vendor/Memora/app/locomo"
for _path in (REPO, REPO / "vendor/Memora/src", APP):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.chdir(APP)  # run_memora.py resolves ./ paths from the app dir

from memobench import runtime  # noqa: E402
from memobench.llm import Req, Result  # noqa: E402


class CaptureMiss(Exception):
    """Raised instead of an API call. An Exception on purpose: search treats it like a failed call."""


class CaptureLLM:
    """Stands in for the shared LLM during capture: cache hits pass through, misses are recorded and raised."""

    def __init__(self, real) -> None:
        self.real = real
        self._lock = threading.Lock()
        self.seen: dict[str, Req] = {}
        self.missed: dict[str, Req] = {}

    def call(self, req: Req) -> Result:
        hit = self.real.lookup(req)
        with self._lock:
            self.seen.setdefault(req.key, req)
            if hit is None:
                self.missed.setdefault(req.key, req)
        if hit is None:
            raise CaptureMiss(f"not cached: stage={req.meta.get('stage')} key={req.key[:12]}")
        return hit


def main(overrides: list[str]) -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(APP / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.general.project_path = str(APP)
    if cfg.retrieval.strategy != "semantic":
        raise SystemExit("only retrieval.strategy=semantic (its retrieval makes no LLM calls)")
    if not os.path.exists(cfg.memory.persist_path):
        raise SystemExit(f"no store at {cfg.memory.persist_path}; build it first")

    from providers.memora.search import MemoraSearch

    real = runtime.get_llm()
    capture = CaptureLLM(real)
    saved = runtime.get_llm
    runtime.get_llm = lambda: capture
    logging.disable(logging.ERROR)  # every miss logs "LLM call failed"; the summary below is what matters
    try:
        with tempfile.TemporaryDirectory() as tmp:
            MemoraSearch(cfg, os.path.join(tmp, "capture_output.json"), cfg.memory.top_k, retrieval_strategy="semantic") \
                .process_data_file(os.path.join(cfg.general.data_path, "locomo10.json"))  # fmt: skip
    finally:
        runtime.get_llm = saved
        logging.disable(logging.NOTSET)

    stages = Counter(r.meta.get("stage") for r in capture.seen.values())
    print(f"[answer-prefetch] captured {len(capture.seen)} requests {dict(stages)}; uncached {len(capture.missed)}", flush=True)
    if capture.missed:
        results = real.run(list(capture.missed.values()), mode="batch")
        print(f"[answer-prefetch] batch: {len(results)} results, {sum(1 for r in results.values() if r.error)} errors", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
