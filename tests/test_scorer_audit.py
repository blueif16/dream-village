"""scorer.audit: the integrity audit (design §6.2): code labels every memory line at every commit from dates and
classes, given the one-time LLM mapping of each distinct line to ledger entries."""

from __future__ import annotations

import re

import pytest

from scorer import audit as A
from scorer.gold import Gold
from tests import scorer_village as V
from tests.scorer_fakes import json_reply, make_llm, user_texts
from tests.scorer_replay import make_replay


@pytest.fixture
def gold(tmp_path):
    return Gold(V.build(tmp_path / "village"))


def m(*entries, kind="fact"):
    """A line mapping: entries are (id, relation, current)."""
    return {"kind": kind, "entries": [{"id": i, "relation": r, "current": c} for i, r, c in entries]}


def label(gold, mapping, t):
    return A.label_line(gold, V.P, mapping, t)


# ------------------------------------------------------------------ labels by code


def test_a_line_becomes_stale_once_the_change_is_told(gold):
    line = m(("kai.f01", "consistent", True))
    assert label(gold, line, "2025-09-04T12:00:00") == "supported"
    assert label(gold, line, "2025-09-06T12:00:00") == "stale"


def test_a_value_presented_as_previous_is_not_stale(gold):
    line = m(("kai.f02", "consistent", True), ("kai.f01", "consistent", False))
    assert label(gold, line, "2025-09-06T12:00:00") == "supported"


def test_a_plan_line_turns_stale_when_its_date_passes(gold):
    line = m(("kai.f06", "consistent", True))
    assert label(gold, line, "2025-09-07T10:00:00") == "supported"
    assert label(gold, line, "2025-09-08T12:00:00") == "stale"


def test_an_impression_is_premature_until_enough_episodes_then_supported(gold):
    line = m(("kai.i01", "consistent", True), kind="trait")
    assert label(gold, line, "2025-09-01T09:00:00") == "premature"  # the commit of s01: one episode shown
    assert label(gold, line, "2025-09-05T19:00:00") == "supported"  # s03's commit: two episodes


def test_a_decoy_trait_is_over_inferred(gold):
    assert label(gold, m(("kai.i02", "consistent", True), kind="trait"), "2025-09-12T00:00:00") == "over_inferred"


def test_an_unmapped_trait_is_over_inferred_and_an_unmapped_fact_fabricated(gold):
    assert label(gold, m(kind="trait"), "2025-09-12T00:00:00") == "over_inferred"
    assert label(gold, m(kind="fact"), "2025-09-12T00:00:00") == "fabricated"


def test_a_true_fact_not_told_yet_is_fabricated(gold):
    line = m(("kai.f07", "consistent", True))
    assert label(gold, line, "2025-09-08T00:00:00") == "fabricated"
    assert label(gold, line, "2025-09-10T09:00:00") == "supported"


def test_never_store_and_forgotten_items_are_leaked(gold):
    assert label(gold, m(("kai.f04", "consistent", True)), "2025-09-06T00:00:00") == "leaked"
    crush = m(("kai.f03", "consistent", True))
    assert label(gold, crush, "2025-09-04T00:00:00") == "supported"
    assert label(gold, crush, "2025-09-07T10:00:00") == "leaked"  # the commit of the session that asked to forget


def test_a_contradicting_line_is_contradicted(gold):
    assert label(gold, m(("kai.f02", "contradicts", True)), "2025-09-06T00:00:00") == "contradicted"


def test_a_line_asserting_several_entries_takes_its_most_severe_label(gold):
    t = "2025-09-06T12:00:00"
    assert label(gold, m(("kai.f02", "consistent", True), ("kai.f04", "consistent", True)), t) == "leaked"
    assert label(gold, m(("kai.f02", "consistent", True), ("kai.f01", "consistent", True)), t) == "stale"


def test_a_non_claim_is_not_labeled(gold):
    assert label(gold, {"kind": "not_a_claim", "entries": []}, "2025-09-06T00:00:00") is None


def test_missed_counts_useful_entries_valid_at_t_with_no_supporting_line(gold):
    missed = A.missed(gold, V.P, "2025-09-12T00:00:00", supported={"kai.f02", "kai.i01"})
    # not f05 (trivia), f03 (forgotten), f04 (never-store), f01/f06 (ended), f07 (its only disclosure was flagged
    # missing by the verifier, so the system may never have seen it)
    assert missed == ["cast.f01", "kai.r01"]


def test_an_entry_supported_by_a_line_is_not_missed(gold):
    assert "kai.r01" not in A.missed(gold, V.P, "2025-09-12T00:00:00", supported={"kai.r01"})


# ------------------------------------------------------------------ lines and the pipeline


def test_memory_lines_skip_frontmatter_blank_lines_and_headings():
    text = "---\nname: food\ndescription: d\n---\n\n# Food\n- [stated] vegetarian (kai.s03)\n\nlikes tea\n"
    assert A.memory_lines({"/topics/food.md": text}) == [("/topics/food.md", "- [stated] vegetarian (kai.s03)"),
                                                         ("/topics/food.md", "likes tea")]  # fmt: skip


def test_each_commit_records_profile_size_and_files_missing_their_frontmatter(gold, tmp_path):
    good = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] tea\n"
    long = "---\nname: x\ndescription: " + "d" * 151 + "\n---\n- [stated] y\n"
    replay = make_replay(tmp_path / "replay", [("2025-09-01T09:00:00", "write", "kai.s01", {
        "/profile.md": "- [stated] one two three\n", "/topics/food.md": good, "/topics/bad.md": "- [stated] no header\n",
        "/areas/long.md": long})])  # fmt: skip
    [c] = A.plan_audit(gold, replay, [V.P])[V.P]["commits"]
    assert c["profile_words"] == 4  # counted as v0 counts its cap: tokens holding a letter or digit ("-" is not)
    assert c["contract_violations"] == ["/areas/long.md", "/topics/bad.md"]


DIET_OLD ="- [stated] eats everything, carnitas forever (kai.s01)"
DIET_NEW = "- [stated] vegetarian (previously eats everything) (kai.s03)"
TRAIT = "- [inferred] downplays her achievements (kai.s01, kai.s03)"


def mapper(params):
    """Maps each numbered line by its text, the way the real matcher would."""
    lines = re.findall(r'<line n="(\d+)" file="[^"]*">\n(.*?)\n</line>', user_texts(params)[-1], re.S)
    table = {DIET_OLD: m(("kai.f01", "consistent", True)),
             DIET_NEW: m(("kai.f02", "consistent", True), ("kai.f01", "consistent", False)),
             TRAIT: m(("kai.i01", "consistent", True), kind="trait")}  # fmt: skip
    return json_reply({"lines": [{"n": int(n), **table[text]} for n, text in lines]})


def test_each_distinct_line_is_mapped_once_and_labeled_at_every_commit(gold, tmp_path):
    replay = make_replay(tmp_path / "replay", [
        ("2025-09-01T09:00:00", "write", "kai.s01", {"/topics/food.md": f"{DIET_OLD}\n", "/profile.md": f"{TRAIT}\n"}),
        ("2025-09-02T03:00:00", "dream", None, {"/topics/food.md": f"{DIET_OLD}\n\n"}),
        ("2025-09-05T19:00:00", "write", "kai.s03", {"/topics/food.md": f"{DIET_OLD}\n{DIET_NEW}\n"}),
    ])  # fmt: skip
    llm, fake = make_llm(tmp_path / "runs", mapper)
    result = A.run_audit(llm, gold, replay, [V.P])
    assert len(fake.seen) == 1  # one request: three distinct lines for one principal
    sent = re.findall(r'<line n="\d+"', user_texts(fake.seen[0])[-1])
    assert len(sent) == 3
    series = result["principals"][V.P]["series"]
    assert [s["counts"] for s in series] == [
        {"premature": 1, "supported": 1},
        {"premature": 1, "supported": 1},
        {"stale": 1, "supported": 2},  # the old diet line is stale once the change is told; the trait now has 2 episodes
    ]
    assert series[-1]["precision"] == pytest.approx(2 / 3)
    assert series[0]["size_words"] > 0
    ledger = user_texts(fake.seen[0])[0]
    assert "kai.i02" in ledger and "kai.f04" in ledger  # every entry can be matched, decoys and withheld ones too
    assert "decoy" not in ledger and "never_store" not in ledger  # the matcher sees content, never classes
    assert result["principals"][V.P]["series"][-1]["missed"] == ["cast.f01", "kai.r01"]


# ------------------------------------------------------------------ the pilot-v2 files (D32 c, e)


@pytest.fixture
def v2(tmp_path):
    return Gold(V.build(tmp_path / "village", v2=True))


def test_a_memory_line_holding_a_planner_minted_never_store_fact_is_leaked(v2):
    assert label(v2, m(("kai.w1.ns01", "consistent", True)), "2025-09-07T10:00:00") == "leaked"


def test_a_memory_line_holding_a_session_fact_is_supported_once_its_session_happened(v2):
    line = m(("kai.s02.t01", "consistent", True))
    assert label(v2, line, "2025-09-03T20:00:00") == "supported"  # the commit of s02's write
    assert label(v2, line, "2025-09-02T00:00:00") == "fabricated"


def test_the_matcher_ledger_lists_the_v2_entries(v2):
    ledger = A.ledger_block(v2, V.P)
    assert "kai.w1.ns01" in ledger and "kai.s02.t01" in ledger and "1998 wall unit" in ledger
