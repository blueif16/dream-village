from __future__ import annotations

import time

import pytest

from memobench.llm import LLM
from tests.fakes import FakeAnthropic


@pytest.fixture
def fake() -> FakeAnthropic:
    return FakeAnthropic()


@pytest.fixture
def sleeps() -> list[float]:
    """Every duration passed to the injected sleep, in order."""
    return []


@pytest.fixture
def make_llm(tmp_path, fake, sleeps):
    """Build an LLM on root=tmp_path with the fake client and a no-op, recording sleep."""

    def _make(**overrides) -> LLM:
        kw = dict(root=str(tmp_path), run_id="r1", client=fake, sleep=sleeps.append)
        kw.update(overrides)
        return LLM(**kw)

    return _make


@pytest.fixture
def llm(make_llm) -> LLM:
    return make_llm()


@pytest.fixture
def tiny_real_sleep(sleeps):
    """Records the requested duration but really sleeps at most 20 ms, so wall-clock time advances
    (needed for timeouts, whether the implementation measures elapsed time by clock or by sleeps)."""

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        time.sleep(max(0.0, min(seconds, 0.02)))

    return _sleep
