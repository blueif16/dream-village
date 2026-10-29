"""memsys.write: the Haiku tool loop that files a finished session into memory (system-v0.md §1 fast write)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from memsys import loop
from memsys.config import V0, load
from memsys.write import write_chat, write_pass
from tests.memsys_fakes import (
    context_text, last_tool_results, refusal_reply, scripted_llm, text_reply, tool_use, tools_reply,
)  # fmt: skip
from village.schema import Session, Turn

FOOD = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] eats chicken (kai.s01, 2025-09-01)\n"


def session(sid: str, start: str, user: str = "I've gone vegetarian since last month.") -> Session:
    return Session(sid, "kai", start, [Turn("user", user), Turn("assistant", "Got it, here are some recipes.")])


S2 = session("kai.s02", "2025-09-10T19:30:00")


@pytest.fixture
def system():
    return load(V0)


@pytest.fixture
def store(system, tmp_path):
    s = system.store(tmp_path / "kai")
    s.write("/topics/food.md", FOOD)
    return s


def with_hops(system, n: int):
    return replace(system, config={**system.config, "max_tool_hops": {**system.config["max_tool_hops"], "write": n}})


def test_write_pass_applies_the_scripted_tool_calls_then_stops_when_the_model_stops(system, store, tmp_path):
    (v,) = [r["version"] for r in store.read(["/topics/food.md"])]
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_read", paths=["/topics/food.md"])),
        tools_reply(
            tool_use("memory_str_replace", path="/topics/food.md", old_str="eats chicken (kai.s01, 2025-09-01)",
                     new_str="vegetarian since 2025-08 (previously ate chicken) (kai.s02, 2025-09-10)", version=v),
            tool_use("memory_append", path="/profile.md", line="- [stated] name is Kai (kai.s02, 2025-09-10)"),
        ),
        text_reply("DONE: 2 edits: /topics/food.md, /profile.md"),
    ])  # fmt: skip
    chat = write_pass(llm, system, store, S2, archive=[S2])
    assert len(client.calls) == 3
    assert "- [stated] vegetarian since 2025-08 (previously ate chicken) (kai.s02, 2025-09-10)" in store.text("/topics/food.md")
    assert store.text("/profile.md") == "- [stated] name is Kai (kai.s02, 2025-09-10)\n"
    assert chat.text == "DONE: 2 edits: /topics/food.md, /profile.md"
    assert chat.edits == 2


def test_write_pass_stops_at_max_tool_hops_when_the_model_keeps_calling_tools(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", lambda params: tools_reply(
        tool_use("memory_append", path="/topics/food.md", line=f"- [stated] line {len(params['messages'])}")))
    write_pass(llm, with_hops(system, 3), store, S2, archive=[S2])
    assert len(client.calls) == 3
    assert store.text("/topics/food.md").count("- [stated] line") == 3  # the last hop's calls are applied too


def test_a_refused_edit_reaches_the_model_as_an_error_result(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_str_replace", path="/topics/food.md", old_str="eats chicken", new_str="vegan",
                             version="000000000000")),
        text_reply("NO CHANGES: stale"),
    ])  # fmt: skip
    write_pass(llm, system, store, S2, archive=[S2])
    (result,) = last_tool_results(client.calls[1])
    assert result["is_error"] is True and "stale" in result["content"]
    assert store.text("/topics/food.md") == FOOD


def test_first_request_has_a_cached_static_system_prompt_and_the_session_in_the_user_turn(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [text_reply("NO CHANGES: nothing durable")])
    write_pass(llm, system, store, S2, archive=[S2])
    params = client.calls[0]
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"] == [{"type": "text", "text": system.prompts["write"], "cache_control": {"type": "ephemeral"}}]
    ctx = context_text(params)
    for part in ("<profile", "<preferences", "<memory_listing>", "/topics/food.md", 'id="kai.s02"', "2025-09-10",
                 "User: I've gone vegetarian since last month.", "Assistant: Got it, here are some recipes."):  # fmt: skip
        assert part in ctx, part
    assert {t["name"] for t in params["tools"]} == {
        "memory_read", "memory_write", "memory_str_replace", "memory_append", "memory_delete", "archive_search"}
    assert "temperature" not in params


def test_each_hop_resends_the_history_unchanged_with_the_assistant_turn_verbatim(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_read", paths=["/topics/food.md"])), text_reply("NO CHANGES: x")])
    write_pass(llm, system, store, S2, archive=[S2])
    first, second = client.calls
    assert second["messages"][0] == first["messages"][0]
    assistant = second["messages"][1]
    assert assistant["role"] == "assistant"
    assert [b["type"] for b in assistant["content"]] == ["thinking", "tool_use"]
    assert assistant["content"][0]["signature"]  # thinking blocks go back unmodified


def test_a_refusal_ends_the_pass_without_raising_and_is_reported(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [refusal_reply()])
    chat = write_pass(llm, system, store, S2, archive=[S2])
    assert chat.refused is True and len(client.calls) == 1
    assert store.text("/topics/food.md") == FOOD


def test_archive_search_in_the_write_pass_sees_only_earlier_sessions(system, store, tmp_path):
    archive = [session("kai.s01", "2025-09-01T08:00:00", "I hate cilantro"), S2,
               session("kai.s03", "2025-09-20T08:00:00", "cilantro again")]  # fmt: skip
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("archive_search", query="Cilantro")), text_reply("NO CHANGES: x")])
    write_pass(llm, system, store, S2, archive=archive)
    (result,) = last_tool_results(client.calls[1])
    assert "kai.s01" in result["content"] and "kai.s03" not in result["content"] and "kai.s02" not in result["content"]


def test_salt_changes_the_cache_key_but_not_the_prompt(system, store):
    plain = loop.step(write_chat(system, store, S2, archive=[S2]))
    salted = loop.step(write_chat(system, store, S2, archive=[S2], salt="a1"))
    assert salted.params["metadata"] == {"user_id": "replay-a1"}
    assert "metadata" not in plain.params
    assert {k: v for k, v in salted.params.items() if k != "metadata"} == plain.params
    assert salted.key != plain.key


def test_write_after_turn_is_refused_before_any_call(system, store, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", [])
    with pytest.raises(NotImplementedError):
        write_pass(llm, replace(system, config={**system.config, "write_after": "turn"}), store, S2, archive=[S2])
    assert client.calls == []
