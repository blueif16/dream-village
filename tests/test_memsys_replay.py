"""memsys.replay: sessions in time order through write passes and nightly dreams, a git commit per state change,
a timeline sidecar, and read-only historical checkouts."""

from __future__ import annotations

import json
import subprocess

import pytest

from memsys.config import V0, load
from memsys.replay import checkout, load_sessions, replay
from tests.memsys_fakes import context_text, refusal_reply, scripted_llm, text_reply, tool_use, tools_reply
from village.schema import Session, Turn, to_dict

FOOD = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] coffee (kai.s1, 2025-09-10)\n"


def session(sid: str, start: str, principal: str = "kai") -> Session:
    return Session(sid, principal, start, [Turn("user", f"hello from {sid}"), Turn("assistant", "hi")])


SESSIONS = [
    session("kai.s1", "2025-09-10T19:00:00"),
    session("kai.s2", "2025-09-10T21:00:00"),
    session("kai.s3", "2025-09-12T08:00:00"),
]


@pytest.fixture
def system():
    return load(V0)


def scripted(system, refuse: tuple[str, ...] = ()):
    """write kai.s1: create food.md; write kai.s2: nothing; write kai.s3: append; dream: append a dated line.
    A session id in `refuse` gets a refusal."""

    def script(params):
        ctx, first_hop = context_text(params), len(params["messages"]) == 1
        if not first_hop:
            return text_reply("DONE")
        if params["system"][0]["text"] == system.prompts["dream"]:
            day = ctx.split("Today is ")[1][:10]
            return tools_reply(tool_use("memory_append", path="/topics/food.md", line=f"- [stated] dreamed {day}"))
        sid = ctx.split('<session id="')[1].split('"')[0]
        if sid.split(".")[1] in refuse or sid in refuse:
            return refusal_reply()
        if sid.endswith(".s1"):
            return tools_reply(tool_use("memory_write", path="/topics/food.md", content=FOOD))
        if sid.endswith(".s3"):
            return tools_reply(tool_use("memory_append", path="/topics/food.md", line=f"- [stated] tea ({sid})"))
        return text_reply("NO CHANGES: nothing durable")

    return script


def timeline(out, principal="kai") -> list[dict]:
    return [json.loads(line) for line in (out / principal / "timeline.jsonl").read_text().splitlines()]


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


# ----------------------------------------------------------------------------- loader


def test_load_sessions_reads_json_and_jsonl_files_in_a_directory_but_not_subdirectories(tmp_path):
    d = tmp_path / "sessions"
    (d / "attempt1").mkdir(parents=True)
    (d / "kai.s1.json").write_text(json.dumps(to_dict(SESSIONS[0])))
    (d / "more.json").write_text(json.dumps([to_dict(SESSIONS[1])]))
    (d / "rest.jsonl").write_text(json.dumps(to_dict(SESSIONS[2])) + "\n")
    (d / "attempt1" / "kai.s1.json").write_text(json.dumps(to_dict(session("kai.s1", "2025-01-01T00:00:00"))))
    got = load_sessions(d)
    assert sorted(s.id for s in got) == ["kai.s1", "kai.s2", "kai.s3"]
    assert next(s for s in got if s.id == "kai.s1").start == "2025-09-10T19:00:00"


# ----------------------------------------------------------------------------- replay + timeline


def test_replay_commits_each_state_change_with_the_sim_time(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    events = [(r["event"], r["session_id"], r["sim_time"]) for r in timeline(out)]
    assert events == [
        ("write", "kai.s1", "2025-09-10T19:00:00"),  # kai.s2 changed nothing: no commit
        ("dream", None, "2025-09-11T03:00:00"),
        ("write", "kai.s3", "2025-09-12T08:00:00"),
        ("dream", None, "2025-09-13T03:00:00"),
    ]
    log = git(out / "kai", "log", "--reverse", "--format=%H %ad %cd", "--date=iso-strict").split("\n")[:-1]
    assert [line.split()[0] for line in log] == [r["sha"] for r in timeline(out)]
    assert [line.split()[1] for line in log] == [r["sim_time"] + "+00:00" for r in timeline(out)]
    assert [line.split()[2] for line in log] == [r["sim_time"] + "+00:00" for r in timeline(out)]


def test_timeline_records_config_hash_models_cost_and_tokens(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    first = timeline(out)[0]
    assert first["config_sha256"] == system.sha256
    assert first["models"] == {"write": "claude-haiku-5-5"}
    assert first["cost_usd"] > 0 and first["tokens"]["input_tokens"] == 2000  # two hops x 1000 (fake usage)
    assert timeline(out)[1]["models"] == {"dream": "claude-haiku-5-5"}


def test_until_stops_the_replay_and_the_trailing_dream(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out, until="2025-09-12")
    assert [r["sim_time"] for r in timeline(out)][-1] == "2025-09-12T08:00:00"


def test_a_refused_write_is_logged_and_the_chain_continues(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system, refuse=("s1",)))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    events = [json.loads(line) for line in (out / "kai" / "events.jsonl").read_text().splitlines()]
    assert events[0]["session_id"] == "kai.s1" and events[0]["refused"] is True and events[0]["sha"] is None
    assert any(e["session_id"] == "kai.s3" for e in events)


def test_salt_is_sent_on_every_request(system, tmp_path):
    llm, client = scripted_llm(tmp_path / "runs", scripted(system))
    replay(llm, system, SESSIONS, tmp_path / "out", salt="aa2")
    assert client.calls and all(c["metadata"] == {"user_id": "replay-aa2"} for c in client.calls)


def test_principals_each_get_their_own_repo_and_the_filter_applies(system, tmp_path):
    sessions = SESSIONS + [session("mo.s1", "2025-09-10T09:00:00", "mo"), session("zed.s1", "2025-09-10T09:00:00", "zed")]
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, sessions, out, principals=["kai", "mo"])
    assert (out / "kai" / ".git").is_dir() and (out / "mo" / ".git").is_dir() and not (out / "zed").exists()
    assert [r["session_id"] for r in timeline(out, "mo")][0] == "mo.s1"


def test_manifest_summarizes_the_run(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    m = json.loads((out / "replay_manifest.json").read_text())
    assert m["config_sha256"] == system.sha256 and m["prompt_sha256"] == system.prompt_sha256
    assert m["models"] == system.models and m["sessions"] == 3 and m["commits"] == 4
    assert m["cost_usd"] == pytest.approx(llm.spent_usd)
    assert m["cost_per_1m_session_tokens_est"] > 0 and m["wall_s"] >= 0


def test_replaying_into_a_used_out_dir_is_refused(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    replay(llm, system, SESSIONS, tmp_path / "out")
    with pytest.raises(FileExistsError):
        replay(llm, system, SESSIONS, tmp_path / "out")


# ----------------------------------------------------------------------------- checkout


def test_checkout_returns_the_memory_as_of_t(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    after_s1 = checkout(out, "kai", "2025-09-10T23:00:00")
    assert (after_s1 / "topics" / "food.md").read_text() == FOOD
    after_dream = checkout(out, "kai", "2025-09-12T07:59:00")
    assert (after_dream / "topics" / "food.md").read_text() == FOOD + "- [stated] dreamed 2025-09-11\n"
    assert after_s1 != after_dream and not (after_s1 / "timeline.jsonl").exists()


def test_checkout_before_the_first_commit_is_an_empty_memory(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    empty = checkout(out, "kai", "2025-09-10T18:59:00")
    assert empty.is_dir() and list(empty.iterdir()) == []


def test_checkout_never_touches_the_timeline_or_the_repo(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    before = (out / "kai" / "timeline.jsonl").read_bytes(), git(out / "kai", "status", "--porcelain"), git(out / "kai", "rev-parse", "HEAD")
    fork = checkout(out, "kai", "2025-09-12T09:00:00")
    (fork / "topics" / "food.md").write_text("changed on the fork")
    after = (out / "kai" / "timeline.jsonl").read_bytes(), git(out / "kai", "status", "--porcelain"), git(out / "kai", "rev-parse", "HEAD")
    assert after == before
    assert "changed on the fork" not in (out / "kai" / "topics" / "food.md").read_text()


def test_checkout_at_exactly_a_commits_sim_time_includes_that_commit(system, tmp_path):
    llm, _ = scripted_llm(tmp_path / "runs", scripted(system))
    out = tmp_path / "out"
    replay(llm, system, SESSIONS, out)
    at_dream = checkout(out, "kai", "2025-09-11T03:00:00")
    assert (at_dream / "topics" / "food.md").read_text() == FOOD + "- [stated] dreamed 2025-09-11\n"


def test_budget_counts_only_this_replays_spend_not_other_runs_on_the_shared_ledger(system, tmp_path):
    # round 1: four candidate replays died when a concurrent scoring run's spend on runs/ledger.jsonl ate their allowance
    from memsys.replay import own_budget

    llm, client = scripted_llm(tmp_path / "runs", scripted(system))
    own_budget(llm, 1.0)
    with open(tmp_path / "runs" / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "other", "stage": "x", "key": "k", "cost_usd": 5.0, "mode": "batch",
                            "ts": "2026-10-28T00:00:00+00:00"}) + "\n")  # fmt: skip
    replay(llm, system, SESSIONS, tmp_path / "out")
    assert client.calls and timeline(tmp_path / "out")
