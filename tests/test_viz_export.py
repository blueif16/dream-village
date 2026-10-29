"""viz/export.py: per-commit memory snapshots and diffs read from a replay git repo.

The fixture is a tiny replay repo built here (three commits: add, modify+add nested, delete) with the
timeline.jsonl sidecar a real replay writes beside it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from viz import export as EX

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]

# The memory files at each commit, written by hand so the expectations are independent of the exporter.
STATES = [
    {"profile.md": "# Dana\nworks nights\n", "topics/food.md": "likes pho\n"},
    {"profile.md": "# Dana\nworks days now\n", "topics/food.md": "likes pho\n", "people/sam.md": "sister\n"},
    {"profile.md": "# Dana\nworks days now\n", "people/sam.md": "sister\n"},
]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def replay(tmp_path: Path) -> Path:
    repo = tmp_path / "dana"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".git" / "info" / "exclude").write_text("timeline.jsonl\n")
    timeline = []
    for k, state in enumerate(STATES):
        for f in repo.rglob("*.md"):
            f.unlink()
        for path, text in state.items():
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"write dana.s0{k + 1}")
        sha = _git(repo, "rev-parse", "HEAD").strip()
        timeline.append({"sha": sha, "sim_time": f"2025-08-2{5 + k}T10:00:00", "event": "write",
                         "session_id": f"dana.s0{k + 1}", "cost_usd": 0.001, "tokens": {"output_tokens": 10}})
    (repo / "timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in timeline))
    return repo


def test_snapshot_at_each_commit_equals_the_files_committed_there(replay: Path) -> None:
    commits = EX.read_commits(replay)
    assert [c["snapshot"] for c in commits] == STATES


def test_diff_lists_exactly_the_files_changed_since_the_previous_commit(replay: Path) -> None:
    commits = EX.read_commits(replay)
    changed = [{(f["path"], f["status"]) for f in c["files"]} for c in commits]
    assert changed == [
        {("profile.md", "A"), ("topics/food.md", "A")},
        {("profile.md", "M"), ("people/sam.md", "A")},
        {("topics/food.md", "D")},
    ]
    second = commits[1]["diff"]
    assert "-works nights" in second and "+works days now" in second and "+sister" in second
    assert "food.md" not in second  # unchanged at commit 2, so absent from its diff


def test_a_day_is_spent_at_the_place_its_life_log_names_else_at_home() -> None:
    places = [
        {"id": "pl-cafe", "name": "Heron & Grind", "kind": "cafe"},
        {"id": "pl-park", "name": "Delgado Park", "kind": "park"},
    ]
    entries = [
        {"date": "2025-08-25", "what": "Dana wakes late and answers email in bed.", "with": []},
        {"date": "2025-08-25", "what": "After class she studies at heron & grind until close.", "with": []},
        {"date": "2025-08-26", "what": "She folds laundry and calls her sister.", "with": []},
    ]
    days = EX.day_positions(entries, ["2025-08-25", "2025-08-26"], places, home="home-dana", anchor="pl-park")
    assert (days["2025-08-25"]["place"], days["2025-08-25"]["why"]) == ("pl-cafe", "life_log")
    assert (days["2025-08-26"]["place"], days["2025-08-26"]["why"]) == ("home-dana", "home")


def test_index_sources_are_repo_relative_so_the_bundle_can_be_published():
    from pathlib import Path

    import viz.export as E

    root = Path(E.__file__).resolve().parent.parent
    assert E._public_path(root / "data" / "villages" / "v1") == "data/villages/v1"
    assert E._public_path(Path("runs/scorer/x")) == "runs/scorer/x"
    assert E._public_path(Path("/elsewhere/abs/dir")) == "dir"
