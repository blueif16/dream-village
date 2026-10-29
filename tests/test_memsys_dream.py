"""memsys.dream: nightly consolidation on the simulated clock (v0: 03:00 local, only after activity)."""

from __future__ import annotations

from datetime import datetime

import pytest

from memsys.config import V0, load
from memsys.dream import dream_due, dream_pass
from tests.memsys_fakes import context_text, last_tool_results, scripted_llm, text_reply, tool_use, tools_reply

FOOD = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] coffee (kai.s01, 2025-09-01)\n"


@pytest.fixture
def system():
    return load(V0)


@pytest.fixture
def store(system, tmp_path):
    s = system.store(tmp_path / "kai")
    s.write("/topics/food.md", FOOD)
    s.write("/profile.md", "- [stated] name is Kai (kai.s01, 2025-09-01)\n")
    return s


def test_dream_is_not_due_before_the_first_0300_after_activity(system):
    assert dream_due(system, "2025-09-10T19:30:00", 1, "2025-09-11T02:59:00") is None


def test_dream_is_due_at_the_first_0300_crossing_after_activity(system):
    assert dream_due(system, "2025-09-10T19:30:00", 1, "2025-09-11T03:00:00") == datetime(2025, 9, 11, 3, 0)
    assert dream_due(system, "2025-09-10T19:30:00", 1, "2025-09-14T10:00:00") == datetime(2025, 9, 11, 3, 0)


def test_activity_after_midnight_dreams_the_same_night(system):
    assert dream_due(system, "2025-09-11T01:00:00", 1, "2025-09-11T08:00:00") == datetime(2025, 9, 11, 3, 0)


def test_no_dream_without_a_session_since_the_last_dream(system):
    assert dream_due(system, "2025-09-10T19:30:00", 0, "2025-09-20T08:00:00") is None
    assert dream_due(system, None, 0, "2025-09-20T08:00:00") is None


def test_dream_context_has_todays_date_and_every_file_body(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [text_reply("NO CHANGES: memory is tidy")])
    dream_pass(llm, system, store, "2025-09-11T03:00:00")
    params = client.calls[0]
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"] == [{"type": "text", "text": system.prompts["dream"], "cache_control": {"type": "ephemeral"}}]
    ctx = context_text(params)
    (food,) = store.read(["/topics/food.md"])
    assert "Today is 2025-09-11" in ctx
    assert f'<file path="/topics/food.md" version="{food["version"]}">' in ctx and "- [stated] coffee (kai.s01" in ctx
    assert "<profile" in ctx and "name is Kai" in ctx and "<memory_listing>" in ctx
    assert "archive_search" not in {t["name"] for t in params["tools"]}


def test_dream_applies_small_edits(system, store, tmp_path):
    (food,) = store.read(["/topics/food.md"])
    llm, _ = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_str_replace", path="/topics/food.md", old_str="- [stated] coffee",
                             new_str="- [stated] tea (previously coffee)", version=food["version"])),
        text_reply("DONE: 1 edits: tea"),
    ])  # fmt: skip
    chat = dream_pass(llm, system, store, "2025-09-11T03:00:00")
    assert "- [stated] tea (previously coffee) (kai.s01, 2025-09-01)" in store.text("/topics/food.md")
    assert chat.edits == 1


def test_dream_may_not_create_a_new_file(system, store, tmp_path):
    new = "---\nname: misc\ndescription: Misc\nsources: [chat]\naliases: []\n---\n\n- [stated] x\n"
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_write", path="/topics/misc.md", content=new)), text_reply("NO CHANGES: x")])
    dream_pass(llm, system, store, "2025-09-11T03:00:00")
    (result,) = last_tool_results(client.calls[1])
    assert result["is_error"] is True
    assert not store.exists("/topics/misc.md")
