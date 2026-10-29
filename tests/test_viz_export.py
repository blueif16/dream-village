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


# ------------------------------------------------------------------ optimization round 1 panel
# Fake candidate scorecards in tmp_path, shaped like the scorer's (scorer/scorecard.py `compare`): `deltas` are
# candidate minus v0 as fractions; `paired` has a matched delta and a CI over principals for the four check
# dimensions only, never integrity. The C4 confirm numbers are D42's (+3.4 [−1, 8], +4.6 [−4, 14], +1.8 [−1, 5],
# integrity +3.5); the screen's `deltas` differ from its `paired` deltas, as a 4-principal screen's do against v0's
# 12-principal scorecard.

R1_PROVISIONAL = "provisional (Claude-only judges, D28)"


def _card(root: Path, rel: str, *, deltas: dict, paired: dict, principals: list[str], probes: int) -> None:
    d = root / rel
    d.mkdir(parents=True)
    card = {
        "eval": {"probes": probes, "provisional": R1_PROVISIONAL, "rubric_version": "dv-judge-v2"},
        "systems": {"v0": {"integrity": {"score": 0.6, "final": {p: {"recall": 0.6} for p in principals}}}},
        "comparison": {"system": "v0", "deltas": deltas, "paired": paired, "flip_share": 0.09, "flip_checks": 1246,
                       "checks_needed": None, "gates": {"integrity_no_rise": {"pass": False, "rises": []}},
                       "provisional": R1_PROVISIONAL, "a": "runs/scorer/full13-v1-v0-evalv2"},
        "costs": {},
    }  # fmt: skip
    (d / "scorecard.json").write_text(json.dumps(card))


TWELVE = [f"p{i}" for i in range(12)]
FOUR = ["arjun", "isabel", "rochelle", "grace"]


@pytest.fixture
def round1(tmp_path: Path) -> tuple[dict, Path]:
    _card(tmp_path, "runs/scorer/confirm-c4", principals=TWELVE, probes=443,
          deltas={"remembering": 0.0339622641509435, "understanding": 0.0461538461538461,
                  "restraint": 0.01773049645390079, "task": -0.0023529411764706687, "integrity": 0.03508400550490187},
          paired={"remembering": {"delta": 0.033962264150943396, "ci_principals": [-0.006921620872165221, 0.07894736842105263], "n": 265},
                  "understanding": {"delta": 0.046153846153846156, "ci_principals": [-0.04371584699453552, 0.13761818947762586], "n": 195},
                  "restraint": {"delta": 0.01773049645390071, "ci_principals": [-0.011494252873563218, 0.04727272727272727], "n": 282},
                  "task": {"delta": -0.002352941176470588, "ci_principals": [-0.007194244604316547, 0.0], "n": 425}})  # fmt: skip
    _card(tmp_path, "runs/scorer/screen-c4", principals=FOUR, probes=146,
          deltas={"remembering": 0.081, "understanding": -0.029, "restraint": 0.022, "task": -0.012, "integrity": 0.037},
          paired={"remembering": {"delta": 0.044, "ci_principals": [0.015, 0.067], "n": 91},
                  "understanding": {"delta": -0.026, "ci_principals": [-0.141, 0.047], "n": 77},
                  "restraint": {"delta": -0.01, "ci_principals": [-0.065, 0.052], "n": 97},
                  "task": {"delta": -0.007, "ci_principals": [-0.021, 0.0], "n": 143}})  # fmt: skip
    _card(tmp_path, "runs/scorer/aa", principals=TWELVE, probes=212,
          deltas={"remembering": 0.012195121951219523, "understanding": 0.01851851851851849,
                  "restraint": 0.021505376344086113, "task": 0.0, "integrity": -0.009816663454863206},
          paired={"remembering": {"delta": 0.012195121951219513, "ci_principals": [-0.03977272727272727, 0.06802721088435375], "n": 164},
                  "understanding": {"delta": 0.018518518518518517, "ci_principals": [-0.03478640592367571, 0.07111111111111111], "n": 216},
                  "restraint": {"delta": 0.021505376344086023, "ci_principals": [-0.022222222222222223, 0.05952380952380952], "n": 93},
                  "task": {"delta": 0.0, "ci_principals": [0.0, 0.0], "n": 208}})  # fmt: skip
    spec = {
        "lesson": "Round 1: no single change beat v0 beyond noise; combining them raised recall but cost integrity.",
        "rule": "Acceptance rule, fixed before the runs: the target gain is beyond the A/A noise and no dimension "
                "drops beyond it.",
        "noise_floor": {"label": "A/A run (D38, eval v1)", "scorecard": "runs/scorer/aa"},
        "candidates": [
            {"id": "C3", "change": "the nightly dream writes `[inferred]` lines", "verdict": "screened out",
             "reason": None, "screen": "runs/scorer/screen-c3", "confirm": None},
            {"id": "C4", "change": "forget masks the forgotten span in the raw window and archive",
             "verdict": "not accepted", "reason": "missed its own target (forgotten-item checks 34/38 vs v0 35/38)",
             "screen": "runs/scorer/screen-c4", "confirm": "runs/scorer/confirm-c4"},
        ],
    }  # fmt: skip
    return spec, tmp_path


def _cand(block: dict, cid: str) -> dict:
    return next(c for c in block["candidates"] if c["id"] == cid)


def test_round1_check_deltas_are_paired_points_with_their_ci_and_integrity_has_no_ci(round1) -> None:
    spec, root = round1
    c4 = _cand(EX.round1_summary(spec, root), "C4")
    assert c4["confirm"]["dims"] == {
        "remembering": {"delta": 3.4, "ci": [-0.7, 7.9], "source": "paired"},
        "understanding": {"delta": 4.6, "ci": [-4.4, 13.8], "source": "paired"},
        "restraint": {"delta": 1.8, "ci": [-1.1, 4.7], "source": "paired"},
        "integrity": {"delta": 3.5, "ci": None, "source": "deltas"},
    }
    # the screen's check dimensions take the matched delta that its CI belongs to, not the unpaired `deltas`
    assert c4["screen"]["dims"]["remembering"] == {"delta": 4.4, "ci": [1.5, 6.7], "source": "paired"}
    assert c4["screen"]["dims"]["integrity"] == {"delta": 3.7, "ci": None, "source": "deltas"}
    assert (c4["screen"]["principals"], c4["screen"]["probes"]) == (sorted(FOUR), 146)
    assert (len(c4["confirm"]["principals"]), c4["confirm"]["probes"]) == (12, 443)


def test_round1_noise_floor_is_read_from_the_aa_scorecard(round1) -> None:
    spec, root = round1
    nf = EX.round1_summary(spec, root)["noise_floor"]
    assert nf["label"] == "A/A run (D38, eval v1)" and nf["present"] is True
    assert {d: (x["delta"], x["ci"]) for d, x in nf["dims"].items()} == {
        "remembering": (1.2, [-4.0, 6.8]), "understanding": (1.9, [-3.5, 7.1]),
        "restraint": (2.2, [-2.2, 6.0]), "integrity": (-1.0, None),
    }  # fmt: skip


def test_round1_missing_scorecard_is_absent_never_filled_in(round1) -> None:
    spec, root = round1
    spec["candidates"][1]["confirm"] = "runs/scorer/confirm-c9"  # never written
    block = EX.round1_summary(spec, root)
    absent = {"scorecard": "runs/scorer/confirm-c9", "present": False, "probes": None, "principals": None,
              "provisional": None, "dims": None}  # fmt: skip
    assert _cand(block, "C4")["confirm"] == absent
    assert _cand(block, "C3")["screen"] == {**absent, "scorecard": "runs/scorer/screen-c3"}  # its dir was never made
    assert _cand(block, "C3")["confirm"] is None  # not in the spec: no confirmation was run


def test_round1_keeps_the_hand_authored_text_and_repo_relative_paths(round1) -> None:
    spec, root = round1
    block = EX.round1_summary(spec, root)
    assert (block["lesson"], block["rule"]) == (spec["lesson"], spec["rule"])
    c4 = _cand(block, "C4")
    assert {k: c4[k] for k in ("id", "change", "verdict", "reason")} == {k: spec["candidates"][1][k]
                                                                         for k in ("id", "change", "verdict", "reason")}
    assert c4["confirm"]["scorecard"] == "runs/scorer/confirm-c4"
    assert c4["confirm"]["provisional"] == R1_PROVISIONAL
    assert str(root) not in json.dumps(block)  # publishable: no local absolute path
