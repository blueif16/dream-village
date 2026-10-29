"""Helpers for tests of the Memora app layer (vendor/Memora/app/{locomo,longmemeval}).

* `use_app(monkeypatch, name)` puts one app dir + Memora's src on sys.path and purges the app-level
  module names (`utils`, `evals`, `metrics`, ...) that both apps define, so each test imports the right app.
  `memora.memora_client` is replaced by a stub: these tests never touch a memory store.
* `RecordingLLM` stands in for `memobench.runtime.get_llm()`: records every `call` / `run`, never hits a network.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import memobench.llm
from memobench.llm import Req, Result

REPO = Path(__file__).resolve().parent.parent
MEMORA = REPO / "vendor" / "Memora"
APPS = MEMORA / "app"
UPSTREAM = REPO / "vendor" / "upstream"
MODEL = "claude-haiku-5-5"

_APP_MODULES = {"utils", "evals", "prompts", "metrics", "providers", "answerer", "run_memora", "run_experiments"}


class StubMemoraClient:
    def __init__(self, cfg, user_id=None, **kw):
        self.user_id = user_id

    def get(self, memory_id):
        return None


def use_app(monkeypatch, name: str):
    for mod in list(sys.modules):
        if mod.split(".")[0] in _APP_MODULES:
            monkeypatch.delitem(sys.modules, mod)
    monkeypatch.syspath_prepend(str(MEMORA / "src"))
    monkeypatch.syspath_prepend(str(APPS / name))
    stub = types.ModuleType("memora.memora_client")
    stub.MemoraClient = StubMemoraClient
    monkeypatch.setitem(sys.modules, "memora.memora_client", stub)
    _ensure_req_key(monkeypatch)
    return APPS / name


def load_cfg(app_dir: Path):
    from omegaconf import OmegaConf

    return OmegaConf.load(app_dir / "conf" / "config.yaml")


def _ensure_req_key(monkeypatch):
    """memobench.llm is built in parallel; if Req.key is still a stub, use the spec's definition."""
    try:
        Req(params={}).key
    except NotImplementedError:
        def key(self):
            blob = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(blob.encode()).hexdigest()

        monkeypatch.setattr(memobench.llm.Req, "key", property(key))


class RecordingLLM:
    """Fake process-wide LLM. `reply(req) -> str` decides each response text."""

    def __init__(self, reply=lambda req: "FINAL ANSWER: stub answer"):
        self.reply = reply
        self.calls: list[Req] = []
        self.runs: list[tuple[list[Req], str, dict]] = []

    def _result(self, req: Req, mode: str) -> Result:
        return Result(key=req.key, text=self.reply(req), message={}, stop_reason="end_turn", usage={},
                      cost_usd=0.0, latency_ms=1.0, cached=False, mode=mode, batch_id=None, error=None)

    def call(self, req: Req) -> Result:
        self.calls.append(req)
        return self._result(req, "sync")

    def run(self, reqs, mode, **kw) -> dict[str, Result]:
        reqs = list(reqs)
        self.runs.append((reqs, mode, kw))
        return {r.key: self._result(r, "batch" if mode == "batch" else "sync") for r in reqs}


def install_llm(monkeypatch, llm: RecordingLLM) -> RecordingLLM:
    import memobench.runtime

    monkeypatch.setattr(memobench.runtime, "get_llm", lambda: llm)
    return llm


def user_text(req: Req) -> str:
    (msg,) = req.params["messages"]
    assert msg["role"] == "user"
    content = msg["content"]
    if isinstance(content, str):
        return content
    assert all(b["type"] == "text" for b in content)
    return "".join(b["text"] for b in content)


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


def tiny_locomo() -> list[dict]:
    """Two LoCoMo-shaped conversations; each has one category-5 question that must never be answered."""

    def conv(sid, a, b, s1, s2, qa):
        return {
            "sample_id": sid,
            "conversation": {
                "speaker_a": a, "speaker_b": b,
                "session_1_date_time": "1:00 pm on 8 May, 2023",
                "session_1": [{"speaker": a, "dia_id": "D1:1", "text": s1[0]}, {"speaker": b, "dia_id": "D1:2", "text": s1[1]}],
                "session_2_date_time": "2:00 pm on 9 May, 2023",
                "session_2": [{"speaker": b, "dia_id": "D2:1", "text": s2}],
            },
            "qa": qa,
        }

    return [
        conv("conv-a", "Ann", "Bob", ("I adopted a cat.", "Nice!"), "I moved to Oslo.", [
            {"question": "What pet did Ann adopt?", "answer": "a cat", "category": 4, "evidence": ["D1:1"]},
            {"question": "Where did Ann move to?", "adversarial_answer": "Oslo", "category": 5, "evidence": ["D2:1"]},
            {"question": "When did Bob move?", "answer": "9 May 2023", "category": 2, "evidence": ["D2:1"]},
        ]),
        conv("conv-b", "Cy", "Di", ("I run marathons.", "Cool."), "I bought a red bike.", [
            {"question": "What colour bike did Cy buy?", "adversarial_answer": "red", "category": 5, "evidence": ["D2:1"]},
            {"question": "What sport does Cy do?", "answer": "running", "category": 1, "evidence": ["D1:1"]},
            {"question": "What did Di buy?", "answer": "a red bike", "category": 4, "evidence": ["D2:1"]},
        ]),
    ]


def import_app_module(name: str):
    return importlib.import_module(name)
