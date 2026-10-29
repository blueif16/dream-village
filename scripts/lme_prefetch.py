"""Prefetch the store-independent work of a LongMemEval Memora build: Batch API (50% off) + bulk embeddings.

Phases (each idempotent and resumable; run in order, separately or together):
  A  capture -> ONE `LLM.run(mode="batch")` of every `episodic` + `extraction` request
  B  capture -> ONE `LLM.run(mode="batch")` of every first-pass `cue` request (needs A cached)
  C  capture -> real `memobench.embed.embed` per kind on the deduplicated texts the build embeds (needs A+B cached)
Phase D is the normal build (`run_memora.py`) with the SAME hydra overrides; it hits both caches. Nothing here
changes a result: the response cache only answers byte-identical requests, so anything not prefetched runs live.

Capture = the real ingestion path (`providers.memora.add.MemoraADD` -> Memora builder) on a throwaway store, with
  - `memobench.runtime.get_llm` -> CaptureLLM: cached Result on a hit (`LLM.lookup`, C6); on a miss it records the
    Req and raises CaptureMiss (never the API). Memora treats that like a failed call: no episode / no cues /
    the session is skipped after add_memory's retries;
  - `memobench.embed.embed` -> record (text, kind) and return random unit vectors, so no consolidation candidate
    reaches the update threshold and no `update` call is attempted.
A crash while a batch is polled is resumed by rerunning the same phase (LLM.run B4 picks up the open manifest).

Usage (overrides = those of the run_memora.py build to prefetch; eval.subset_idx is required):
  MEMOBENCH_RUN_ID=<id>-prefetch uv run python scripts/lme_prefetch.py --phases ABC \
      'eval.subset_idx="2,3,4"' memory.update_score_threshold=0.9291 retrieval.strategy=prompt
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "vendor" / "Memora" / "app" / "longmemeval"
for _path in (REPO, REPO / "vendor" / "Memora" / "src", APP):  # as run_memora.py, plus the app dir itself
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import memobench.embed as mb_embed  # noqa: E402
from memobench import runtime  # noqa: E402
from memobench.llm import Req, Result  # noqa: E402

BATCH_STAGES = {"A": ("episodic", "extraction"), "B": ("cue",)}
DATA_FILE = "longmemeval_s_cleaned.json"


class CaptureMiss(Exception):
    """Raised instead of an API call. An Exception on purpose: Memora's own handlers treat it like a failed call."""


class CaptureLLM:
    """Stands in for the shared LLM during capture: cache hits pass through, misses are recorded and raised."""

    def __init__(self, real) -> None:
        self.real = real
        self._lock = threading.Lock()
        self.seen: dict[str, Req] = {}  # every request the build made, by key
        self.missed: dict[str, Req] = {}  # the uncached ones

    def call(self, req: Req) -> Result:
        hit = self.real.lookup(req)
        with self._lock:
            self.seen.setdefault(req.key, req)
            if hit is None:
                self.missed.setdefault(req.key, req)
        if hit is None:
            raise CaptureMiss(f"not cached: stage={req.meta.get('stage')} key={req.key[:12]}")
        return hit


class EmbedRecorder:
    """Stands in for memobench.embed.embed during capture: records (text, kind), returns random unit vectors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pairs: dict[tuple[str, str], None] = {}  # insertion-ordered set
        self.calls = 0

    def __call__(self, texts, kind):
        texts = list(texts)
        with self._lock:
            self.calls += 1
            for t in texts:
                self.pairs.setdefault((t, kind))
        v = np.random.default_rng().standard_normal((len(texts), mb_embed.DIMS))
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()


def compose_cfg(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(APP / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    cfg.general.project_path = str(APP)  # run_memora.py resolves ./ from its launch dir (the app dir)
    return cfg


def capture(cfg, real, work: Path, label: str) -> tuple[CaptureLLM, EmbedRecorder]:
    """One pass of the real ingestion path over the subset, on a throwaway store, with both stand-ins installed."""
    from providers.memora.add import MemoraADD

    ccfg = copy.deepcopy(cfg)
    store = Path(tempfile.mkdtemp(prefix=f"capture-{label}-", dir=work))
    ccfg.memory.persist_path = str(store)  # never a real store, whatever the overrides say
    ccfg.memory.force_rebuild = False
    ccfg.memory.skip_existing = False
    ccfg.eval.max_workers = max(int(ccfg.eval.get("max_workers", 10)), 24)  # capture makes no API calls
    llm, rec = CaptureLLM(real), EmbedRecorder()
    import providers.memora.add as lme_add

    class _NoSleepTime:  # add_memory sleeps 1 s between retries; a CaptureMiss looks like a failure to it
        def __getattr__(self, name):
            return (lambda *_a, **_k: None) if name == "sleep" else getattr(time, name)

    saved_time = lme_add.time

    saved_get_llm, saved_embed = runtime.get_llm, mb_embed.embed
    saved_root = os.environ.get("MEMOBENCH_ROOT")
    root_logger = logging.getLogger()
    saved_handlers, saved_level = root_logger.handlers[:], root_logger.level
    t0 = time.monotonic()
    log_path = work / f"capture_{label}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            handler = logging.StreamHandler(log)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            root_logger.handlers, root_logger.level = [handler], logging.WARNING
            runtime.get_llm = lambda: llm
            mb_embed.embed = rec
            lme_add.time = _NoSleepTime()
            os.environ["MEMOBENCH_ROOT"] = str(work)  # capture-time parse-failure lines stay out of the shared log
            MemoraADD(ccfg, data_path=str(Path(ccfg.general.data_path) / DATA_FILE)).process_all_conversations()
    finally:
        runtime.get_llm, mb_embed.embed = saved_get_llm, saved_embed
        lme_add.time = saved_time
        if saved_root is None:
            os.environ.pop("MEMOBENCH_ROOT", None)
        else:
            os.environ["MEMOBENCH_ROOT"] = saved_root
        root_logger.handlers, root_logger.level = saved_handlers, saved_level
        shutil.rmtree(store, ignore_errors=True)

    hits = Counter(r.meta.get("stage") for k, r in llm.seen.items() if k not in llm.missed)
    misses = Counter(r.meta.get("stage") for r in llm.missed.values())
    print(
        f"[prefetch] capture {label}: {time.monotonic() - t0:.0f}s  cached={dict(hits)}  uncached={dict(misses)}  "
        f"embed calls={rec.calls} unique (text, kind)={len(rec.pairs)}  log={log_path}",
        flush=True,
    )
    return llm, rec


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batch_phase(phase: str, cfg, real, work: Path, approved: bool) -> None:
    stages = BATCH_STAGES[phase]
    llm, _ = capture(cfg, real, work, phase)
    reqs = [r for r in llm.missed.values() if r.meta.get("stage") in stages]
    write_jsonl(  # every first-pass request of this phase, to diff against the build's live requests
        work / f"phase_{phase}_reqs.jsonl",
        ({"key": k, "cached": k not in llm.missed, "meta": r.meta, "params": r.params}
         for k, r in llm.seen.items() if r.meta.get("stage") in stages),
    )
    if phase == "B":
        upstream = Counter(r.meta.get("stage") for r in llm.missed.values() if r.meta.get("stage") in BATCH_STAGES["A"])
        if upstream:
            print(f"[prefetch] WARNING: {dict(upstream)} still uncached (phase A incomplete); their cues are not captured")
    if not reqs:
        print(f"[prefetch] phase {phase}: every {'/'.join(stages)} request is cached; nothing to submit", flush=True)
        return
    results = real.run(reqs, mode="batch", approved=approved)
    errors = Counter(r.meta.get("stage") for r in reqs if results[r.key].error)
    print(f"[prefetch] phase {phase}: {len(reqs)} requests batched, errors={dict(errors)} (errors run live in the build)")


def embed_phase(cfg, real, work: Path) -> None:
    llm, rec = capture(cfg, real, work, "C")
    if llm.missed:
        stages = dict(Counter(r.meta.get("stage") for r in llm.missed.values()))
        print(f"[prefetch] WARNING: {stages} still uncached; texts behind them are not captured (run phases A, B)")
    by_kind: dict[str, list[str]] = {}
    for text, kind in rec.pairs:
        by_kind.setdefault(kind, []).append(text)
    write_jsonl(work / "phase_C_embeds.jsonl", ({"kind": k, "text": t} for t, k in rec.pairs))
    t0 = time.monotonic()
    for kind, texts in by_kind.items():
        mb_embed.embed(texts, kind)  # the real one: cache-first, <=100 texts per request
    counts = {k: len(v) for k, v in by_kind.items()}
    print(f"[prefetch] phase C: embedded {counts} in {time.monotonic() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phases", default="ABC", help="any of A, B, C, run in that order (default ABC)")
    ap.add_argument("--work-dir", help="capture logs + captured requests (default <runs>/prefetch/<subset>)")
    ap.add_argument("--approved", action="store_true", help="pass approved=True to LLM.run (estimate > approval)")
    ap.add_argument("overrides", nargs="*", help="run_memora.py hydra overrides; eval.subset_idx=... is required")
    args = ap.parse_args(argv)
    phases = args.phases.upper()
    if not phases or set(phases) - set("ABC"):
        ap.error("--phases takes letters from A, B, C")
    if not any(o.startswith("eval.subset_idx=") for o in args.overrides):
        ap.error("pass eval.subset_idx=... : the prefetch covers exactly the subset you will build")

    cfg = compose_cfg(args.overrides)
    real = runtime.get_llm()  # created before capture swaps MEMOBENCH_ROOT
    subset = re.sub(r"[^0-9A-Za-z]+", "_", str(cfg.eval.subset_idx)).strip("_")
    work = Path(args.work_dir) if args.work_dir else Path(real.root) / "prefetch" / f"lme_{subset}"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[prefetch] run_id={real.run_id} subset={cfg.eval.subset_idx} phases={phases} work={work}", flush=True)

    for phase in "ABC":
        if phase not in phases:
            continue
        if phase == "C":
            embed_phase(cfg, real, work)
        else:
            batch_phase(phase, cfg, real, work, args.approved)


if __name__ == "__main__":
    main()
