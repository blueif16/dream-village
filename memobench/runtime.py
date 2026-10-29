"""Process-wide LLM instance, configured from env (.env at the repo root)."""
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

# Backbone for every answer, judge and memory-system call (baseline-protocol.md). Default: the frozen Haiku;
# MEMOBENCH_MODEL=gemini-4-argon is the Argon rerun (same prompts, different model -> different cache keys).
MODEL = os.environ.get("MEMOBENCH_MODEL", "claude-haiku-5-5")

_lock = threading.Lock()
_llm = None


def get_llm():
    global _llm
    with _lock:
        if _llm is None:
            from memobench.llm import LLM

            _llm = LLM(
                root=os.environ.get("MEMOBENCH_ROOT", str(REPO / "runs")),
                run_id=os.environ.get("MEMOBENCH_RUN_ID", "dev"),
                budget_usd=float(os.environ.get("MEMOBENCH_BUDGET_USD", "10")),
                max_attempts=int(os.environ.get("MEMOBENCH_MAX_ATTEMPTS", "6")),
                approval_usd=float(os.environ.get("MEMOBENCH_APPROVAL_USD", "50")),
                allowed_models=(MODEL,),  # one backbone per process: an Argon run can never call Claude
                gemini_concurrency=int(os.environ.get("MEMOBENCH_GEMINI_CONCURRENCY", "64")),
            )
        return _llm
