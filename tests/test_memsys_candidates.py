"""Optimization round 1 candidates (T5): each config is ONE change versus v0, and v0 stays byte-identical.

C1 answer reads memory first · C2 file on first mention + profile every session · C3 dream writes [inferred] ·
C4 forget redacts the raw window and archive · C5 all four.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memsys.answer import answer_sync
from memsys.config import V0, load
from memsys.dream import dream_pass
from memsys.replay import checkout, replay
from memsys.write import write_pass
from tests.memsys_fakes import context_text, last_tool_results, scripted_llm, text_reply, tool_use, tools_reply
from village.schema import Session, Turn

CONFIGS = Path(__file__).resolve().parent.parent / "memsys" / "config"
V0_SHA256 = "1950477b514ab2be24c85b09057b9d3b881855f2002066ba7e8dbac09988ae0d"  # runs/replays/full13-v1-v0 manifest
MASK = "[removed at the user's request]"


def cand(n: int):
    return load(CONFIGS / f"c{n}.json")


def session(sid: str, start: str, user: str, assistant: str = "Okay.") -> Session:
    return Session(sid, "kai", start, [Turn("user", user), Turn("assistant", assistant)])


S1 = session("kai.s01", "2025-09-01T08:00:00", "I owe my cousin 400 dollars and it stresses me out.",
             "Owing family 400 dollars can feel heavy.")  # fmt: skip
S2 = session("kai.s02", "2025-09-05T08:00:00", "Please forget what I said about owing my cousin money.")
S3 = session("kai.s03", "2025-09-08T08:00:00", "Any budgeting tips?")
SESSIONS = [S1, S2, S3]


# ----------------------------------------------------------------------------- configs


def test_v0_system_hash_is_unchanged():
    assert load(V0).sha256 == V0_SHA256


@pytest.mark.parametrize(
    "n, steps, keys",
    [
        (1, {"answer"}, {"answer_read_policy"}),
        (2, {"write"}, {"write_threshold", "profile_refresh"}),
        (3, {"dream"}, {"dream_inference"}),
        (4, {"write"}, {"forget"}),
        (5, {"answer", "write", "dream"}, {"answer_read_policy", "write_threshold", "profile_refresh",
                                           "dream_inference", "forget"}),
    ],
)  # fmt: skip
def test_each_candidate_changes_only_its_settings_and_prompts(n, steps, keys):
    v0, c = load(V0), cand(n)
    assert {s for s in v0.prompts if c.prompts[s] != v0.prompts[s]} == steps
    changed = {k for k in set(c.config) | set(v0.config) if c.config.get(k) != v0.config.get(k)}
    assert changed - {"prompts", "system"} == keys
    assert c.sha256 != v0.sha256


def test_c2_selects_the_first_mention_write_prompt():
    c2 = cand(2)
    assert c2.config["write_threshold"] == "first_mention"
    assert c2.prompts["write"] == (CONFIGS.parent / "prompts" / "write_c2.md").read_text()


def test_a_setting_that_disagrees_with_its_prompt_is_refused(tmp_path):
    cfg = json.loads(V0.read_text())
    cfg["write_threshold"] = "first_mention"  # but the prompt is still v0's write.md
    cfg["prompts"] = {k: str(CONFIGS / v) for k, v in cfg["prompts"].items()}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="write_threshold"):
        load(path)


def test_a_variant_prompt_without_its_setting_is_refused(tmp_path):
    cfg = json.loads(V0.read_text())
    cfg["prompts"] = {k: str(CONFIGS / v) for k, v in cfg["prompts"].items()}
    cfg["prompts"]["dream"] = str(CONFIGS.parent / "prompts" / "dream_c3.md")  # but dream_inference is absent (v0)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="dream_inference"):
        load(path)


# ----------------------------------------------------------------------------- C3: the dream can search the archive


def test_c3_dream_searches_only_sessions_before_the_dream(tmp_path):
    c3 = cand(3)
    store = c3.store(tmp_path / "kai")
    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("archive_search", query="budgeting cousin")), text_reply("NO CHANGES: memory is tidy")])
    dream_pass(llm, c3, store, "2025-09-06T03:00:00", archive=SESSIONS)
    assert "archive_search" in {t["name"] for t in client.calls[0]["tools"]}
    (result,) = last_tool_results(client.calls[1])
    assert "kai.s01" in result["content"] and "kai.s03" not in result["content"]


def test_c3_dream_may_create_a_topics_file_for_inferred_lines_but_nothing_else(tmp_path):
    c3 = cand(3)
    store = c3.store(tmp_path / "kai")

    def fm(name):
        return f"---\nname: {name}\ndescription: x\nsources: [chat]\naliases: []\n---\n\n- [inferred] y (evidence: kai.s01, kai.s03)\n"

    llm, client = scripted_llm(tmp_path / "runs", [
        tools_reply(tool_use("memory_write", path="/topics/habits.md", content=fm("habits")),
                    tool_use("memory_write", path="/people/cousin.md", content=fm("cousin"))),
        text_reply("DONE: 1 edits"),
    ])  # fmt: skip
    dream_pass(llm, c3, store, "2025-09-06T03:00:00", archive=SESSIONS)
    ok, refused = last_tool_results(client.calls[1])
    assert ok["is_error"] is False and refused["is_error"] is True
    assert store.exists("/topics/habits.md") and not store.exists("/people/cousin.md")


# ----------------------------------------------------------------------------- C4: forget redacts the raw history


def forget_script(quote: str, sid: str = "kai.s01"):
    return [tools_reply(tool_use("forget_span", session_id=sid, quote=quote)), text_reply("DONE: 1 edits")]


def test_c4_forget_span_records_a_sidecar_that_is_not_a_memory_file(tmp_path):
    c4 = cand(4)
    store = c4.store(tmp_path / "kai")
    llm, client = scripted_llm(tmp_path / "runs", forget_script("I owe my cousin 400 dollars"))
    write_pass(llm, c4, store, S2, archive=SESSIONS)
    assert "forget_span" in {t["name"] for t in client.calls[0]["tools"]}
    (result,) = last_tool_results(client.calls[1])
    assert result["is_error"] is False
    assert store.files() == [] and "(no memory files yet)" in store.listing()
    recorded = [json.loads(line) for line in (store.root / ".system" / "forgotten.jsonl").read_text().splitlines()]
    assert [(r["session_id"], r["quote"]) for r in recorded] == [("kai.s01", "I owe my cousin 400 dollars")]


def test_c4_forget_span_with_a_quote_not_in_that_session_is_refused(tmp_path):
    c4 = cand(4)
    store = c4.store(tmp_path / "kai")
    llm, client = scripted_llm(tmp_path / "runs", forget_script("I owe my brother"))
    write_pass(llm, c4, store, S2, archive=SESSIONS)
    (result,) = last_tool_results(client.calls[1])
    assert result["is_error"] is True
    assert not (store.root / ".system" / "forgotten.jsonl").exists()


def test_v0_offers_no_forget_span(tmp_path):
    v0 = load(V0)
    llm, client = scripted_llm(tmp_path / "runs", [text_reply("NO CHANGES: x")])
    write_pass(llm, v0, v0.store(tmp_path / "kai"), S2, archive=SESSIONS)
    assert "forget_span" not in {t["name"] for t in client.calls[0]["tools"]}


def test_c4_forgotten_spans_are_masked_in_the_recent_window_and_archive_search(tmp_path):
    c4 = cand(4)
    store = c4.store(tmp_path / "kai")
    llm, _ = scripted_llm(tmp_path / "runs", forget_script("I owe my cousin 400 dollars"))
    write_pass(llm, c4, store, S2, archive=SESSIONS)
    llm, client = scripted_llm(tmp_path / "runs2", [
        tools_reply(tool_use("archive_search", query="cousin")), text_reply("Here are some tips.")])
    answer_sync(llm, c4, store.root, SESSIONS, "Any budgeting tips?", "2025-09-10T08:00:00")
    ctx = context_text(client.calls[0])
    assert "I owe my cousin 400 dollars" not in ctx and MASK in ctx
    assert "and it stresses me out" in ctx  # only the quoted span goes
    (result,) = last_tool_results(client.calls[1])
    assert "I owe my cousin 400 dollars" not in result["content"]


def test_c4_masking_reaches_probes_through_the_replay_checkout(tmp_path):
    c4 = cand(4)

    def script(params):
        if len(params["messages"]) > 1:
            return text_reply("DONE")
        if 'id="kai.s02"' in context_text(params):
            return tools_reply(tool_use("forget_span", session_id="kai.s01", quote="i owe my COUSIN 400 dollars"))
        return text_reply("NO CHANGES: x")

    llm, _ = scripted_llm(tmp_path / "runs", script)
    out = tmp_path / "out"
    replay(llm, c4, SESSIONS, out)
    before_forget, after_forget = checkout(out, "kai", "2025-09-04T00:00:00"), checkout(out, "kai", "2025-09-06T00:00:00")
    seen = []
    for fork in (before_forget, after_forget):
        llm, client = scripted_llm(tmp_path / f"runs-{fork.name}", [text_reply("ok")])
        answer_sync(llm, c4, fork, SESSIONS, "tips?", "2025-09-10T08:00:00")
        seen.append(context_text(client.calls[0]))
    assert "I owe my cousin 400 dollars" in seen[0]  # before the forget request, nothing is masked
    assert "I owe my cousin 400 dollars" not in seen[1] and MASK in seen[1]


def test_c4_later_write_passes_search_the_archive_with_the_span_masked(tmp_path):
    c4 = cand(4)
    store = c4.store(tmp_path / "kai")
    llm, _ = scripted_llm(tmp_path / "runs", forget_script("I owe my cousin 400 dollars"))
    write_pass(llm, c4, store, S2, archive=SESSIONS)
    llm, client = scripted_llm(tmp_path / "runs2", [
        tools_reply(tool_use("archive_search", query="dollars")), text_reply("NO CHANGES: x")])  # s01 still matches
    write_pass(llm, c4, store, S3, archive=SESSIONS)
    (result,) = last_tool_results(client.calls[1])
    assert "kai.s01" in result["content"] and "I owe my cousin 400 dollars" not in result["content"]


def test_c4_forget_span_cannot_target_a_later_session(tmp_path):
    c4 = cand(4)
    store = c4.store(tmp_path / "kai")
    llm, client = scripted_llm(tmp_path / "runs", forget_script("Any budgeting tips", sid="kai.s03"))
    write_pass(llm, c4, store, S2, archive=SESSIONS)
    (result,) = last_tool_results(client.calls[1])
    assert result["is_error"] is True
    assert not (store.root / ".system" / "forgotten.jsonl").exists()
