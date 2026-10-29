"""memsys.answer: answering a request at time t, as a resumable state machine (step/feed), read-only on memory."""

from __future__ import annotations

from dataclasses import replace

import pytest

from memobench.llm import Req
from memsys.answer import Final, answer_state, answer_sync, feed, step
from memsys.archive import archive_search
from memsys.config import V0, load
from tests.memsys_fakes import (
    context_text, last_tool_results, refusal_reply, scripted_llm, text_reply, tool_use, tools_reply,
)  # fmt: skip
from village.schema import Session, Turn

FOOD = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] vegetarian since 2025-08 (kai.s02, 2025-09-10)\n"
T = "2025-09-20T12:00:00"


def session(sid: str, start: str, user: str) -> Session:
    return Session(sid, "kai", start, [Turn("user", user), Turn("assistant", "Sure.")])


SESSIONS = [
    session("kai.s01", "2025-09-01T08:00:00", "My landlord ignored the leak again."),
    session("kai.s02", "2025-09-10T19:30:00", "Dinner ideas please, I'm vegetarian."),
    session("kai.s03", "2025-09-15T09:00:00", "Help me write to my LANDLORD about the deposit."),
    session("kai.s04", T, "Landlord news: he fixed it."),  # at t: not yet in the past
    session("kai.s05", "2025-09-25T09:00:00", "The landlord called."),
]


@pytest.fixture
def system():
    return load(V0)


@pytest.fixture
def memory(system, tmp_path):
    s = system.store(tmp_path / "kai")
    s.write("/topics/food.md", FOOD)
    s.write("/preferences.md", "- [stated] short answers (kai.s01, 2025-09-01)\n")
    return s


def snapshot(root) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# ----------------------------------------------------------------------------- archive_search


def test_archive_search_excludes_sessions_at_or_after_t():
    found = archive_search(SESSIONS, T, "landlord")
    assert "kai.s01" in found and "kai.s03" in found
    assert "kai.s04" not in found and "kai.s05" not in found


def test_archive_search_is_case_insensitive_and_returns_id_date_and_snippet():
    found = archive_search(SESSIONS, T, "Deposit")
    assert 'id="kai.s03"' in found and "2025-09-15" in found and "the deposit" in found
    assert "kai.s01" not in found


def test_archive_search_with_no_match_says_so():
    assert "No past conversations match" in archive_search(SESSIONS, T, "zebra")


# ----------------------------------------------------------------------------- state machine


def test_answer_round_trips_through_step_and_feed(system, memory, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_read", paths=["/topics/food.md"])), text_reply("Try a lentil curry.")])
    state = answer_state(system, memory.root, SESSIONS, "What should I cook tonight?", T)
    first = step(state)
    assert isinstance(first, Req)
    feed(state, llm.call(first))
    second = step(state)
    assert isinstance(second, Req)
    (result,) = last_tool_results(second.params)
    assert "vegetarian since 2025-08" in result["content"]
    feed(state, llm.call(second))
    final = step(state)
    assert isinstance(final, Final)
    assert final.text == "Try a lentil curry." and final.hops == 2 and not final.refused
    assert step(state) == final  # done stays done


def test_answer_sync_drives_the_same_loop(system, memory, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", [text_reply("Hello!")])
    final = answer_sync(llm, system, memory.root, SESSIONS, "hi", T)
    assert final.text == "Hello!" and final.hops == 1


def test_answering_never_changes_memory(system, memory, tmp_path):
    before = snapshot(memory.root)
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_append", path="/topics/food.md", line="- [stated] vegan"),
                    tool_use("memory_delete", path="/topics/food.md")),
        text_reply("ok"),
    ])  # fmt: skip
    answer_sync(llm, system, memory.root, SESSIONS, "I'm vegan now", T)
    assert {t["name"] for t in client.calls[0]["tools"]} == {"memory_read", "archive_search"}
    assert all(r["is_error"] for r in last_tool_results(client.calls[1]))
    assert snapshot(memory.root) == before


def test_context_holds_the_recent_window_before_t_and_the_request(system, memory, tmp_path):
    small = replace(system, config={**system.config, "recent_window_sessions": 2})
    llm, client = scripted_llm(tmp_path / "runs", [text_reply("ok")])
    answer_sync(llm, small, memory.root, SESSIONS, "Any update on the landlord thing?", T)
    params = client.calls[0]
    assert params["model"] == "claude-sonnet-5-5"
    assert params["system"] == [{"type": "text", "text": system.prompts["answer"], "cache_control": {"type": "ephemeral"}}]
    ctx = context_text(params)
    assert 'id="kai.s02"' in ctx and 'id="kai.s03"' in ctx
    assert ctx.index('id="kai.s02"') < ctx.index('id="kai.s03"')  # oldest first
    assert 'id="kai.s01"' not in ctx and 'id="kai.s04"' not in ctx and 'id="kai.s05"' not in ctx
    assert "short answers" in ctx and "<memory_listing>" in ctx and "/topics/food.md" in ctx
    assert "Today is 2025-09-20" in ctx and "Any update on the landlord thing?" in ctx


def test_after_max_hops_the_last_call_forbids_tools_and_returns_an_answer(system, memory, tmp_path):
    two = replace(system, config={**system.config, "max_tool_hops": {**system.config["max_tool_hops"], "answer": 2}})

    def script(params):
        if params.get("tool_choice") == {"type": "none"}:
            return text_reply("Best I can tell: lentils.")
        return tools_reply(tool_use("archive_search", query="dinner"))

    llm, client = scripted_llm(tmp_path / "runs", script)
    final = answer_sync(llm, two, memory.root, SESSIONS, "dinner?", T)
    assert len(client.calls) == 3
    assert "tool_choice" not in client.calls[0] and "tool_choice" not in client.calls[1]
    assert client.calls[2]["tool_choice"] == {"type": "none"}
    assert final.text == "Best I can tell: lentils."


def test_a_refusal_is_a_final_answer_marked_refused(system, memory, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", [refusal_reply()])
    final = answer_sync(llm, system, memory.root, SESSIONS, "hi", T)
    assert final.refused is True and final.text == ""


def test_the_answer_archive_search_tool_sees_only_sessions_before_t(system, memory, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("archive_search", query="landlord")), text_reply("He was slow to fix the leak.")])
    answer_sync(llm, system, memory.root, SESSIONS, "Any update on the landlord thing?", T)
    (result,) = last_tool_results(client.calls[1])
    assert "kai.s01" in result["content"] and "kai.s03" in result["content"]
    assert "kai.s04" not in result["content"] and "kai.s05" not in result["content"]
