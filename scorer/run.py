"""Score a village replay end to end (design §5-§7).

  python -m scorer.run --village <dir> --replay <dir> --out <dir> [--stages probes,answers,judge,audit,score]
                       [--systems v0,no_memory,recent_window,oracle,full_context] [--budget USD]
                       [--principals a,b] [--probes-per-checkpoint 5] [--max-checkpoints N]
                       [--full-context-per-stratum 1] [--probes-from <out dir>] [--compare <out dir>] [--salt S]

Stages, each one Batch round (answers: one round per tool hop) and resumable through the response cache (re-running a
stage rebuilds the same requests, so finished ones are cache hits):
  probes   checkpoints + Opus 5.5 generation + code validation -> probes.jsonl, probes_log.json
           (--probes-from copies another run's probes and shortcut flags instead: A/A runs score the same probes,
           and a v0-only A/A run keeps run 1's shortcut filter)
  answers  every (probe, system) by Sonnet 5.5 -> answers.jsonl (full_context on a stratified subset)
  judge    Opus 5.5 primary + Sonnet 5.5 veto -> verdicts.jsonl; the shortcut filter -> shortcuts.json
  audit    the integrity audit of the --replay timeline -> audit.json
  score    scorecard.json + scorecard.md (+ the A/A comparison with --compare) and cards/ for human labels
Before each paid stage it prints the estimate and refuses (exit 2) if it would exceed what --budget leaves. The LLM
root is the main repository's runs/ (shared ledger) unless --root says otherwise; --budget is this invocation's
allowance on top of that ledger's total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from memobench.llm import LLM
from memsys.config import V0
from scorer import answers as AN
from scorer import audit as AU
from scorer import cards as CA
from scorer import judge as JU
from scorer import probes as PR
from scorer import scorecard as SC
from scorer.batch import upper_usd
from scorer.gold import Gold

MODELS = (PR.MODEL, AN.MODEL, JU.PRIMARY, JU.SECOND, AU.MODEL)
STAGES = ("probes", "answers", "judge", "audit", "score")
V0_HOPS = 2  # v0 answer hops for the answer-stage estimate (pilot-v1 measured mean 1.6)


class Refused(Exception):
    pass


def main_root() -> Path:
    """The main checkout (also from a worktree): the parent of git's common dir."""
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        return Path(common).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        return here.parent


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


class _Budget:
    """--budget is this invocation's allowance. It is charged with this LLM instance's own paid spend only, because
    other runs (village generation, replays) write to the same shared ledger meanwhile; before each stage the LLM's
    ledger-wide hard stop is moved to the ledger total + what is left of the allowance."""

    def __init__(self, llm: LLM, allowance: float) -> None:
        self.llm, self.allowance, self.start = llm, allowance, llm.spent_usd
        llm.budget_usd = llm.total_usd + allowance

    def guard(self, stage: str, est: float, n: int) -> None:
        llm = self.llm
        left = self.allowance - (llm.spent_usd - self.start)
        print(f"[scorer] {stage}: {n} requests, estimate ${est:.4f}; this run spent ${llm.spent_usd - self.start:.4f}, "
              f"budget left ${left:.4f}", flush=True)  # fmt: skip
        if est > left:
            raise Refused(f"{stage}: estimate ${est:.4f} exceeds the ${left:.4f} left of --budget; shrink the stage "
                          "(fewer probes or checkpoints) or raise --budget")  # fmt: skip
        llm.budget_usd = llm.total_usd + left


class _Costs:
    def __init__(self, out: Path, llm: LLM) -> None:
        self.path, self.llm = out / "costs.json", llm
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def add(self, stage: str, before: float, **extra) -> None:
        row = self.data.setdefault(stage, {"usd": 0.0})
        row["usd"] += self.llm.total_usd - before
        row.update(extra)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


def run(args: argparse.Namespace, llm: LLM) -> int:
    budget = _Budget(llm, args.budget)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gold = Gold(args.village, observed=args.observed_disclosures)
    principals = args.principals.split(",") if args.principals else gold.principals
    systems = args.systems.split(",")
    stages = args.stages.split(",")
    unknown = set(stages) - set(STAGES) | set(systems) - set(AN.SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown stages/systems: {sorted(unknown)}")
    costs = _Costs(out, llm)

    if "probes" in stages:
        before = llm.total_usd
        if args.probes_from:
            for name in ("probes.jsonl", "probes_log.json", "shortcuts.json"):  # the shortcut filter travels too
                if (Path(args.probes_from) / name).exists():
                    shutil.copy(Path(args.probes_from) / name, out / name)
            print(f"[scorer] probes: copied from {args.probes_from}")
        else:
            plan_log: list[str] = []
            planned = PR.plan_requests(gold, principals, per_checkpoint=args.probes_per_checkpoint,
                                       max_checkpoints=args.max_checkpoints, log=plan_log)  # fmt: skip
            reqs = [r for _, r in planned]
            est = upper_usd(llm, reqs, PR.OUT_TOKENS)
            budget.guard("probes", est, len(reqs))
            results = llm.run(reqs, mode="batch", poll_s=20, estimate_usd=est)
            g = PR.collect(gold, planned, results)
            _jsonl(out / "probes.jsonl", [p.to_json() for p in g.probes])
            slots = sum(len(r.meta["kinds"]) for _, r in planned)
            dropped = [x for x in g.log if ": dropped," in x]
            (out / "probes_log.json").write_text(json.dumps(
                {"eval_version": PR.EVAL_VERSION, "buckets": g.buckets, "plan_log": plan_log,
                 "checkpoints": [c.id for c, _ in planned], "slots": slots, "made": len(g.probes),
                 "dropped": dropped, "repairs": [x for x in g.log if x not in dropped], "failed": g.failed,
                 "prompt_sha256": hashlib.sha256(PR.PROMPT.read_bytes()).hexdigest()}, indent=2) + "\n")  # fmt: skip
            print(f"[scorer] probes: {len(g.probes)} made from {slots} slots; {len(dropped)} dropped, "
                  f"{len(g.failed)} checkpoints failed")  # fmt: skip
            costs.add("probes", before, requests=len(reqs), probes=len(g.probes))
    probes = [PR.probe_from_json(d) for d in _read_jsonl(out / "probes.jsonl") if d["principal"] in principals]

    if "answers" in stages:
        before = llm.total_usd
        subset = AN.stratified_subset(probes, args.full_context_per_stratum) if "full_context" in systems else set()
        if "full_context" in systems:
            every = AN.first_round(gold, probes, ["full_context"], full_context_ids={p.id for p in probes})
            print(f"[scorer] full_context: all {len(probes)} probes would cost ${upper_usd(llm, every, AN.OUT_TOKENS):.4f};"
                  f" running the stratified subset of {len(subset)}")  # fmt: skip
        first = AN.first_round(gold, probes, systems, replay=args.replay, config=args.config, full_context_ids=subset,
                               salt=args.salt)  # fmt: skip
        v0 = [r for r in first if not str(r.meta.get("system", "")).startswith("scorer-")]  # memsys' own requests
        rest = [r for r in first if str(r.meta.get("system", "")).startswith("scorer-")]
        est = upper_usd(llm, rest, AN.OUT_TOKENS) + V0_HOPS * upper_usd(llm, v0, AN.OUT_TOKENS)
        budget.guard("answers", est, len(first))
        answers = AN.run_answers(llm, gold, probes, systems, replay=args.replay, config=args.config,
                                 full_context_ids=subset, salt=args.salt)  # fmt: skip
        _jsonl(out / "answers.jsonl", answers)
        costs.add("answers", before, answers=len(answers),
                  by_system={s: sum(a["cost_usd"] for a in answers if a["system"] == s) for s in systems})  # fmt: skip
    answers = _read_jsonl(out / "answers.jsonl")

    if "judge" in stages:
        before = llm.total_usd
        reqs = JU.planned_requests(gold, probes, answers)
        est = upper_usd(llm, reqs, JU.OUT_TOKENS)
        budget.guard("judge", est, len(reqs))
        verdicts = JU.run_judges(llm, gold, probes, answers)
        _jsonl(out / "verdicts.jsonl", verdicts)
        no_memory: dict[str, list[dict]] = {}
        for v in verdicts:
            if v["system"] == "no_memory":
                no_memory.setdefault(v["probe_id"], []).append(v)
        if no_memory or not (out / "shortcuts.json").exists():
            flagged = sorted(PR.shortcut_flags(probes, no_memory)) if no_memory else []
            (out / "shortcuts.json").write_text(json.dumps({"flagged": flagged, "had_no_memory": bool(no_memory)}) + "\n")
        else:  # no no_memory answers here (an A/A run of v0): keep the flags copied with --probes-from
            flagged = json.loads((out / "shortcuts.json").read_text())["flagged"]
        print(f"[scorer] judge: {len(verdicts)} verdicts; {len(flagged)} probes flagged as shortcuts")
        costs.add("judge", before, requests=len(reqs), verdicts=len(verdicts))
    verdicts = _read_jsonl(out / "verdicts.jsonl")

    if "audit" in stages:
        if not args.replay:
            raise SystemExit("the audit stage needs --replay")
        before = llm.total_usd
        reqs = AU.planned_requests(gold, args.replay, principals)
        est = upper_usd(llm, reqs, AU.OUT_TOKENS)
        budget.guard("audit", est, len(reqs))
        result = AU.run_audit(llm, gold, args.replay, principals)
        (out / "audit.json").write_text(json.dumps(result, indent=1) + "\n")
        costs.add("audit", before, requests=len(reqs),
                  distinct_lines=sum(a["distinct_lines"] for a in result["principals"].values()))  # fmt: skip

    if "score" in stages:
        shortcuts = json.loads((out / "shortcuts.json").read_text())["flagged"] if (out / "shortcuts.json").exists() else []
        for p in probes:
            p.shortcut = p.id in shortcuts
        audits = {"v0": json.loads((out / "audit.json").read_text())} if (out / "audit.json").exists() else {}
        log = json.loads((out / "probes_log.json").read_text()) if (out / "probes_log.json").exists() else {}
        meta = {"eval_version": sorted({p.eval_version for p in probes}), "rubric_version": JU.RUBRIC_VERSION, "judge_prompt_sha256": JU.prompt_sha256(),
                "judges": {"primary": JU.PRIMARY, "second": JU.SECOND}, "answerer": AN.MODEL,
                "probe_generator": PR.MODEL, "audit_matcher": AU.MODEL,
                "probes_made": log.get("made"), "probe_slots": log.get("slots"),
                "probes_dropped": len(log.get("dropped", [])), "checkpoints_failed": log.get("failed"),
                "village": str(args.village), "replay": str(args.replay),
                "observed_disclosures": args.observed_disclosures}  # fmt: skip
        card = SC.build(gold, probes, verdicts, answers, audits=audits,
                        replays={"v0": Path(args.replay)} if args.replay else {}, eval_meta=meta)  # fmt: skip
        comparison = None
        if args.compare:
            other = Path(args.compare)
            a_card = json.loads((other / "scorecard.json").read_text())
            comparison = SC.compare(a_card, card, _read_jsonl(other / "verdicts.jsonl"), verdicts, "v0",
                                    gold=gold, probes=probes)  # fmt: skip
            card["comparison"] = {**comparison, "a": str(other)}
        card["costs"] = costs.data
        (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
        (out / "scorecard.md").write_text(SC.markdown(card, comparison))
        cards = CA.answer_cards(gold, probes, verdicts, answers)
        for a in audits.values():
            cards += CA.fabricated_cards(gold, a)
        counts = CA.export(out / "cards", CA.stratify(cards))
        print(f"[scorer] score: wrote {out / 'scorecard.md'}; cards {counts}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m scorer.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--village", required=True)
    ap.add_argument("--replay", help="the v0 replay directory (memsys.replay --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--systems", default=",".join(AN.SYSTEMS))
    ap.add_argument("--budget", type=float, default=2.0, help="USD this invocation may spend (default 2)")
    ap.add_argument("--principals", help="comma-separated principal ids (default: all)")
    ap.add_argument("--probes-per-checkpoint", type=int, default=5)
    ap.add_argument("--max-checkpoints", type=int, help="per principal (default: all)")
    ap.add_argument("--full-context-per-stratum", type=int, default=1, help="full_context probes per (principal, kind)")
    ap.add_argument("--probes-from", help="reuse another run's probes.jsonl (A/A runs)")
    ap.add_argument("--compare", help="another run's out dir: report this run minus that one")
    ap.add_argument("--config", default=str(V0), help="the memsys config v0 answers with (a candidate's, e.g. c1.json)")
    ap.add_argument("--salt", help="answer cache salt (metadata.user_id): re-samples answers, same prompts")
    ap.add_argument("--root", help="LLM root (default: <main checkout>/runs, the shared ledger)")
    ap.add_argument("--observed-disclosures", action="store_true",
                    help="also count the verifier's unplanned mentions as disclosures (pilot-v1, before D32 (e))")
    return ap


def main(argv: list[str] | None = None, *, llm: LLM | None = None) -> int:
    args = parser().parse_args(argv)
    if llm is None:
        from dotenv import load_dotenv

        root = main_root()
        load_dotenv(root / ".env")
        llm = LLM(root=args.root or str(root / "runs"), run_id=f"scorer-{Path(args.out).name}", allowed_models=MODELS)
    try:
        return run(args, llm)
    except Refused as e:
        print(f"[scorer] REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
