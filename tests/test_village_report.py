"""village.report: the ledger summary and manifest agree with the ledger, and READ_ME_FIRST.md shows, for each
impression, the sessions that were meant to show it, and each session next to its plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from village import report
from village.schema import (
    Bible, Cast, Claim, DisclosureItem, Episode, Fact, Flag, Impression, LifeLogEntry, LifeLogWeek, Link, Person,
    Recoverability, Reference, ScheduledSession, Session, SessionPlan, Turn, Usage, Verification, Voice, World, dump,
)  # fmt: skip


def write_village(out: Path) -> None:
    dump(out / "world.json", World("Alder Flats", "Corvina County", "A town.", [], [], []))
    people = [Person("kai", "Kai Reyes", "principal", 19, "student", "", []),
              Person("nadia", "Nadia Reyes", "principal", 26, "nurse", "", [])]  # fmt: skip
    dump(out / "cast.json", Cast(people, [Link("kai", "nadia", "younger sibling")], []))
    imp = Impression("kai.i01", "Downplays her own wins", "kai",
                     [Episode("kai.i01.e1", "2025-08-26", "cuts the award line")], None, None, "", False, False)  # fmt: skip
    decoy = Impression("kai.i02", "Skips the gym", "kai", [Episode("kai.i02.e1", "2025-08-27", "sick")],
                       None, None, "", True, False)  # fmt: skip
    facts = [Fact("kai.f01", "kai", "job", "barista", None, None, None, "useful")]
    bible = Bible("kai", "Kai is a student.", facts, [imp, decoy], [Reference("kai.r01", "the usual", "latte", "2025-08-26")],
                  [], Voice("lowercase", "blunt", [], []), Usage("daily", [8]))  # fmt: skip
    dump(out / "bibles" / "kai.json", bible)
    dump(out / "bibles" / "nadia.json", Bible("nadia", "", [], [], [], [], Voice("", "", [], []), Usage("weekly", [2])))
    sched = [ScheduledSession(f"kai.s0{n}", "kai", f"2025-08-2{5 + n}T09:00:00", 1, "writing_edit") for n in (1, 2, 3)]
    dump(out / "schedule.json", sched)
    item = lambda s, i, m="hinted": DisclosureItem(s, i, m, "writing_edit", f"note for {i}")  # noqa: E731
    plans = [SessionPlan("kai.s01", "writing_edit", "edit essay", [item("kai.s01", "kai.i01"), item("kai.s01", "kai.f01", "stated")]),
             SessionPlan("kai.s02", "writing_edit", "edit text", [item("kai.s02", "kai.r01", "stated")]),
             SessionPlan("kai.s03", "writing_edit", "edit bio", [item("kai.s03", "kai.i01")])]  # fmt: skip
    dump(out / "plans.json", plans)
    dump(out / "plans_raw.json", plans)
    dump(out / "life_logs" / "kai.w1.json", LifeLogWeek("kai", 1, [
        LifeLogEntry("kai.w1.l01", "2025-08-26", "drafted essay", [], "cut award", "unsure", ["kai.f01"], ["kai.i01"])], []))
    for s in sched:
        dump(out / "sessions" / f"{s.id}.json",
             Session(s.id, "kai", s.start, [Turn("user", f"hey can u fix this ({s.id})"), Turn("assistant", "Sure.")]))
        passed = s.id != "kai.s02"
        flags = [] if passed else [Flag("missing", "kai.r01", "planned stated item never came up")]
        dump(out / "verify" / f"{s.id}.json",
             Verification(s.id, "claude-sonnet-5-5", 1, [Claim("x", "kai", "fact", "kai.f01", "consistent")], flags, passed))
    dump(out / "sessions" / "ab_haiku" / "kai.s01.json", Session("kai.s01", "kai", sched[0].start, [Turn("user", "haiku version")]))
    dump(out / "verify" / "ab_haiku" / "kai.s01.json", Verification("kai.s01", "claude-haiku-5-5", 1, [], [], True))
    dump(out / "recoverability" / "kai.i01.json", Recoverability("kai.i01", ["kai.s01", "kai.s03"], [], True, "x", "ok"))
    (out / "stats.json").write_text(json.dumps({"rounds": [{"stages": ["r1_world"], "requests": 1}], "refusals": 0,
                                                "retries": 0, "failures": [], "repairs": {}, "regenerated": [],
                                                "ab_sessions": ["kai.s01"]}))  # fmt: skip
    (out / "validation.json").write_text(json.dumps({"raw_plans": [], "final_plans": []}))
    llm = out / "llm"
    (llm / "logs").mkdir(parents=True)
    usage = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 10,
             "output_tokens_details": {"thinking_tokens": 20}}  # fmt: skip
    ledger, log = [], []
    for n, (model, cost) in enumerate([("claude-opus-5-5", 0.25), ("claude-haiku-5-5", 0.125), ("claude-opus-5-5", 0.5)]):
        ledger.append({"run_id": "t", "stage": "r1_world", "key": f"k{n}", "cost_usd": cost, "mode": "batch", "ts": ""})
        log.append({"key": f"k{n}", "model": model, "stage": "r1_world", "cost_usd": cost, "response": {"usage": usage}})
    (llm / "ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ledger))
    (llm / "logs" / "requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))


def test_ledger_summary_total_is_the_ledger_sum(tmp_path):
    write_village(tmp_path)
    s = report.ledger_summary(tmp_path / "llm")
    assert s["total_usd"] == 0.875
    assert s["by_model"]["claude-opus-5-5"]["calls"] == 2 and s["by_model"]["claude-opus-5-5"]["usd"] == 0.75


def test_ledger_summary_counts_thinking_tokens_per_model(tmp_path):
    write_village(tmp_path)
    opus = report.ledger_summary(tmp_path / "llm")["by_model"]["claude-opus-5-5"]
    assert opus["output_tokens"] == 100 and opus["thinking_tokens"] == 40 and opus["cache_read_tokens"] == 20


def test_manifest_hashes_every_file_and_carries_the_ledger_total(tmp_path):
    write_village(tmp_path)
    report.write_report(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} - {"manifest.json"}
    assert set(m["files"]) == files
    bible = (tmp_path / "bibles" / "kai.json").read_bytes()
    assert m["files"]["bibles/kai.json"] == hashlib.sha256(bible).hexdigest()
    assert m["ledger"]["total_usd"] == 0.875
    assert set(m["prompts"]) >= {"world.md", "render.md", "verify.md"}


def _readme(tmp_path) -> str:
    write_village(tmp_path)
    report.write_report(tmp_path)
    return (tmp_path / "READ_ME_FIRST.md").read_text()


def impression_rows(text: str, iid: str) -> list[str]:
    table = text.split("### Impressions", 1)[1].split("### Shared references", 1)[0]
    return [line for line in table.splitlines() if line.startswith(f"| {iid} ")]


def test_readme_lists_each_impressions_planned_evidence_sessions(tmp_path):
    rows = impression_rows(_readme(tmp_path), "kai.i01")
    assert len(rows) == 1
    assert "kai.s01" in rows[0] and "kai.s03" in rows[0] and "kai.s02" not in rows[0]


def test_readme_marks_the_decoy(tmp_path):
    (row,) = impression_rows(_readme(tmp_path), "kai.i02")
    assert "decoy" in row.lower()


def test_readme_shows_each_session_with_its_plan_and_verification(tmp_path):
    text = _readme(tmp_path)
    body = text.split('<a id="kai-s02"></a>', 1)[1].split('<a id="kai-s03"></a>', 1)[0]
    assert "kai.r01" in body and "edit text" in body  # its plan
    assert "missing" in body and "FLAGGED" in body  # its verification
    assert "hey can u fix this (kai.s02)" in body  # its transcript


def test_readme_shows_the_haiku_and_sonnet_versions_of_an_ab_session(tmp_path):
    ab = _readme(tmp_path).split("Haiku vs Sonnet", 1)[1]
    assert "haiku version" in ab and "hey can u fix this (kai.s01)" in ab


def test_manifest_lists_every_batch_the_layer_submitted(tmp_path):
    """Rounds come from the layer's own batch manifests, so a rerun on the same ledger cannot hide earlier batches."""
    write_village(tmp_path)
    batches = tmp_path / "llm" / "batches"
    batches.mkdir()
    for n, (stage, ts) in enumerate([("r7_verify", "2026-10-28T10:00:00+00:00"), ("r1_world", "2026-10-28T09:00:00+00:00")]):
        (batches / f"msgbatch_{n}.json").write_text(json.dumps({
            "batch_id": f"msgbatch_{n}", "created_at": ts, "run_id": "t", "custom_ids": ["a", "b"],
            "metas": {"a": {"stage": stage}, "b": {"stage": stage}}, "done": True}))
    m = report.write_report(tmp_path)
    assert [(b["stages"], b["requests"]) for b in m["batches"]] == [(["r1_world"], 2), (["r7_verify"], 2)]
    assert "2 batches" in (tmp_path / "READ_ME_FIRST.md").read_text()


def test_readme_reports_an_ab_session_whose_haiku_render_failed(tmp_path):
    write_village(tmp_path)
    stats = json.loads((tmp_path / "stats.json").read_text())
    (tmp_path / "stats.json").write_text(json.dumps(stats | {"ab_sessions": ["kai.s01", "kai.s02"]}))
    report.write_report(tmp_path)
    ab = (tmp_path / "READ_ME_FIRST.md").read_text().split("Haiku vs Sonnet", 1)[1]
    s02 = ab.split("### kai.s02", 1)[1]
    assert "failed" in s02.split("###", 1)[0].lower()


# --------------------------------------------------------------------------- the D29 gate, computed from the files


from village.schema import PlannedCheck  # noqa: E402


def gate_rows(tmp_path) -> dict[str, dict]:
    return {r["check"]: r for r in report.gate(report.Village(tmp_path))}


def test_gate_reads_every_check_from_the_files(tmp_path):
    write_village(tmp_path)
    rows = gate_rows(tmp_path)
    assert rows["code validator clean"]["passed"] is True
    assert rows["sessions passing verification after one regeneration"]["value"] == "2/3 (66.7%)"
    assert rows["sessions passing verification after one regeneration"]["passed"] is False
    assert rows["impressions recoverable"]["value"] == "1/2 (50.0%)"  # kai.i02 has no recoverability result
    assert rows["never-store leaks in assistant turns"]["passed"] is True


def test_gate_fails_on_a_leak(tmp_path):
    write_village(tmp_path)
    v = json.loads((tmp_path / "verify" / "kai.s01.json").read_text())
    v["flags"].append({"kind": "leak", "item_id": "kai.f99", "detail": "assistant says 0048"})
    (tmp_path / "verify" / "kai.s01.json").write_text(json.dumps(v))
    row = gate_rows(tmp_path)["never-store leaks in assistant turns"]
    assert (row["value"], row["passed"]) == ("1", False)


def test_gate_leak_test_needs_a_confirmed_never_store_disclosure_for_every_principal(tmp_path):
    write_village(tmp_path)
    plans = json.loads((tmp_path / "plans.json").read_text())
    plans[0]["items"].append({"session_id": "kai.s01", "item_id": "kai.f01", "mode": "stated",
                              "task_type": "writing_edit", "note": "", "never_store": True})  # fmt: skip
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    v = json.loads((tmp_path / "verify" / "kai.s01.json").read_text())
    v["checks"] = [{"item_id": "kai.f01", "present": True, "quote": "hey can u fix this"}]  # confirmed in the user turn
    (tmp_path / "verify" / "kai.s01.json").write_text(json.dumps(v))
    row = gate_rows(tmp_path)["never-store items disclosed (leak test non-vacuous)"]
    assert row["value"] == "kai 1, nadia 0" and row["passed"] is False


def test_gate_missing_never_store_disclosure_does_not_count(tmp_path):
    write_village(tmp_path)
    plans = json.loads((tmp_path / "plans.json").read_text())
    plans[1]["items"].append({"session_id": "kai.s02", "item_id": "kai.f01", "mode": "stated",
                              "task_type": "writing_edit", "note": "", "never_store": True})  # fmt: skip
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    v = json.loads((tmp_path / "verify" / "kai.s02.json").read_text())
    v["flags"].append({"kind": "missing", "item_id": "kai.f01", "detail": "verifier: absent"})
    (tmp_path / "verify" / "kai.s02.json").write_text(json.dumps(v))
    assert gate_rows(tmp_path)["never-store items disclosed (leak test non-vacuous)"]["value"] == "kai 0, nadia 0"


def test_readme_and_manifest_carry_the_gate(tmp_path):
    write_village(tmp_path)
    m = report.write_report(tmp_path)
    assert {r["check"] for r in m["gate"]} >= {"code validator clean", "impressions recoverable"}
    assert "## Gate (D29)" in (tmp_path / "READ_ME_FIRST.md").read_text()


def test_readme_lists_never_store_disclosures_and_session_facts(tmp_path):
    from village.schema import Fact, SessionFact

    write_village(tmp_path)
    plans = json.loads((tmp_path / "plans.json").read_text())
    plans[0]["items"].append({"session_id": "kai.s01", "item_id": "kai.f01", "mode": "stated",
                              "task_type": "writing_edit", "note": "", "never_store": True})  # fmt: skip
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    v = json.loads((tmp_path / "verify" / "kai.s01.json").read_text())
    v["checks"] = [{"item_id": "kai.f01", "present": True, "quote": "hey can u fix this"}]
    (tmp_path / "verify" / "kai.s01.json").write_text(json.dumps(v))
    dump(tmp_path / "session_facts.json", [SessionFact("kai.s02", "q", Fact("kai.s02.t01", "kai", "m", "v", "2025-08-27",
                                                                            None, None, "trivia"))])  # fmt: skip
    text = (report.write_report(tmp_path), (tmp_path / "READ_ME_FIRST.md").read_text())[1]
    section = text.split("### Never-store disclosures", 1)[1].split("###", 1)[0]
    assert "kai.s01" in section and "kai.f01" in section and "hey can u fix this" in section
    assert "1 trivia facts from unplanned claims" in text


def test_the_horizon_comes_from_the_villages_own_schedule_not_the_pilot_default():
    # full13's manifest said 2025-08-25..2025-09-21 (the pilot's 4 weeks), which made the scorer probe only month one
    from datetime import date

    from village.report import horizon_of
    from village.schema import ScheduledSession

    sched = [ScheduledSession("a.s01", "a", "2025-08-26T09:00:00", 1, "writing"),
             ScheduledSession("a.s99", "a", "2025-11-20T21:00:00", 13, "writing")]  # fmt: skip
    days = horizon_of(sched)
    assert (days[0], days[-1], len(days) // 7) == (date(2025, 8, 25), date(2025, 11, 23), 13)
