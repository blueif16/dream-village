"""T2 dry run: build the requests a village can build without spending (R1; every later stage needs an earlier
stage's output), price them with LLM.estimate, and extrapolate every stage from measured pilot token profiles.
$0: no request is sent.

  python -m village.estimate --village full --measured data/villages/pilot-v1 data/villages/pilot-v2 \\
      --out data/villages/full-dry-run

Per-request tokens are the pilot's averages for the stage, scaled by GROWTH: how much bigger a request's prompt
(input, cache write, cache read alike) and its output get in the target village (a 12-month world and cast in every
prefix, bigger bibles, longer evidence lists). Low/high are those factors' ranges; regeneration is a fraction of
sessions re-rendered and re-verified.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from memobench.llm import LLM
from memobench.pricing import cost_usd
from village.generate import MODELS, STAGES, r1_request, upper_bound_usd
from village.spec import VILLAGES, VillageSpec

SESSIONS_PER_WEEK_MEAN = {"daily": 6.5, "several_per_week": 3.5, "weekly": 1.0}  # generate.SESSIONS_PER_WEEK midpoints

# (input low, input high, output low, output high) per request, target village vs the pilot (assumptions, see doc).
GROWTH = {
    "pilot": {},
    "full": {
        "r1_world": (8, 12, 3, 7.4),  # a 364-day calendar in; 50-70 events and 24-36 places out (max 32k)
        "r2_cast": (3, 4, 4, 8),  # the bigger world in; ~100 people and ~300 facts out (max 64k)
        "r3_bible": (3.5, 4.5, 2.5, 4),  # world + cast in; a year of facts, impressions and arcs out
        "r4_life_log": (3, 4, 1, 1.2),  # world + cast + bible prefix; a week out, as in the pilot
        "r5_plan": (3.5, 6, 1, 1.2),  # same prefix plus the other principals' facts
        "r6_render": (3, 4, 1, 1.2),  # same prefix; one session out
        "r7_verify": (3, 8, 1, 1.5),  # the dated fact list grows with the cast and the year
        "r8_recover": (2, 3, 1, 1.5),  # 6-10 evidence sessions instead of 3-4
        "r8_judge": (1, 1.5, 1, 1),
    },
    "full13": {  # FULL's cast (same size), a 13-week world and bibles (~1.5x the pilot's)
        "r1_world": (3, 4, 1.5, 3),  # a 91-day calendar in; 15-25 events out
        "r2_cast": (1.5, 2, 4, 7),  # the smaller world in; the same ~100 people out
        "r3_bible": (2.5, 3.5, 1.5, 2.5),
        "r4_life_log": (2.2, 3, 1, 1.2),
        "r5_plan": (2.5, 4.5, 1, 1.2),
        "r6_render": (2.2, 3, 1, 1.2),
        "r7_verify": (2.5, 5, 1, 1.5),
        "r8_recover": (1.5, 2, 1, 1.5),
        "r8_judge": (1, 1.5, 1, 1),
    },
}


@dataclass
class Profile:
    stage: str
    model: str
    requests: int
    input: float
    cache_write: float
    cache_read: float
    output: float


def profiles(roots: list[Path], since: str | None = None) -> dict[str, Profile]:
    """Average tokens per request by stage from each root's logs/requests.jsonl; a later root overrides a stage.
    `since` (ISO timestamp) drops earlier requests from the LAST root only (e.g. a superseded checkpoint on its
    ledger); earlier roots are other runs and keep every request."""
    out: dict[str, Profile] = {}
    for n, root in enumerate(roots):
        cutoff = since if n == len(roots) - 1 else None
        rows: dict[str, list[dict]] = defaultdict(list)
        path = Path(root) / "logs" / "requests.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            if line.strip():
                r = json.loads(line)
                if cutoff is None or r.get("ts", "") >= cutoff:
                    rows[r["stage"]].append(r)
        for stage, rs in rows.items():
            u = [(r.get("response") or {}).get("usage") or {} for r in rs]
            mean = lambda k: statistics.fmean((x.get(k) or 0) for x in u)  # noqa: E731
            model = Counter(r["model"] for r in rs).most_common(1)[0][0]
            out[stage] = Profile(stage, model, len(rs), mean("input_tokens"), mean("cache_creation_input_tokens"),
                                 mean("cache_read_input_tokens"), mean("output_tokens"))  # fmt: skip
    return out


def counts(spec: VillageSpec) -> dict[str, int]:
    """Requests per stage (first attempts), with impressions at the middle of the bible bar's range."""
    n = len(spec.principals)
    sessions = round(spec.weeks * sum(SESSIONS_PER_WEEK_MEAN[p.frequency] for p in spec.principals))
    impressions = round(n * sum(spec.bible_bar.impressions) / 2)
    return {"r1_world": 1, "r2_cast": 1, "r3_bible": n, "r4_life_log": n * spec.weeks, "r5_plan": n * spec.weeks,
            "r6_render": sessions, "r7_verify": sessions, "r8_recover": impressions, "r8_judge": impressions}  # fmt: skip


def _cost(p: Profile, model: str, k_in: float, k_out: float) -> float:
    usage = {"input_tokens": p.input * k_in, "cache_creation_input_tokens": p.cache_write * k_in,
             "cache_read_input_tokens": p.cache_read * k_in, "output_tokens": p.output * k_out}  # fmt: skip
    return cost_usd(usage, batch=True, model=model)


def dry_run(spec: VillageSpec, measured: list[Path], *, root: Path, client=None, regen: tuple[float, float],
            verifier: str | None = None, stages_only: list[str] | None = None, since: str | None = None) -> dict:  # fmt: skip
    """The T2 estimate. `regen`: (low, high) fraction of sessions regenerated once (re-rendered and re-verified)."""
    llm = LLM(root=str(root), run_id=f"dry-run-{spec.name}", client=client, allowed_models=tuple(MODELS.values()))
    req = r1_request(spec)
    buildable = {"r1_world": {"layer_estimate_usd": round(llm.estimate([req], batch=True), 4),
                              "upper_bound_usd": round(upper_bound_usd([req]), 4), "max_tokens": req.params["max_tokens"],
                              "prompt_chars": len(json.dumps(req.params))}}  # fmt: skip
    prof, n, growth = profiles(measured, since), counts(spec), GROWTH.get(spec.name, {})
    imp_lo = len(spec.principals) * spec.bible_bar.impressions[0]
    imp_hi = len(spec.principals) * spec.bible_bar.impressions[1]
    rows = []

    def add(label: str, stage: str, req_lo: float, req_hi: float) -> None:
        p = prof[stage]
        model = verifier if (stage == "r7_verify" and verifier) else STAGES[stage].model
        k_in_lo, k_in_hi, k_out_lo, k_out_hi = growth.get(stage, (1, 1, 1, 1))
        lo, hi = _cost(p, model, k_in_lo, k_out_lo), _cost(p, model, k_in_hi, k_out_hi)
        rows.append({"stage": label, "model": model, "requests": round(req_lo) if req_lo == req_hi else [round(req_lo), round(req_hi)],
                     "usd_per_request": [round(lo, 5), round(hi, 5)], "low_usd": round(lo * req_lo, 4),
                     "high_usd": round(hi * req_hi, 4)})  # fmt: skip

    for stage in stages_only or list(STAGES):
        if stage in ("r8_recover", "r8_judge") and spec.name != "pilot":
            add(stage, stage, imp_lo, imp_hi)
        else:
            add(stage, stage, n[stage], n[stage])
    if not stages_only:
        add("r6_render (regeneration)", "r6_render", n["r6_render"] * regen[0], n["r6_render"] * regen[1])
        add("r7_verify (re-verify)", "r7_verify", n["r7_verify"] * regen[0], n["r7_verify"] * regen[1])
    total = {"low_usd": round(sum(r["low_usd"] for r in rows), 2), "high_usd": round(sum(r["high_usd"] for r in rows), 2)}
    return {"village": spec.name, "counts": n, "regen": list(regen), "buildable": buildable, "stages": rows,
            "total": total, "profiles": {k: asdict(v) for k, v in prof.items()}, "r1_request": req.params}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m village.estimate")
    ap.add_argument("--village", choices=sorted(VILLAGES), default="full")
    ap.add_argument("--measured", nargs="+", required=True, help="village dirs whose llm/ logs give the token profiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regen", nargs=2, type=float, default=None, help="low/high fraction of sessions regenerated")
    ap.add_argument("--verifier", choices=["haiku", "sonnet"], default=None)
    ap.add_argument("--since", default=None, help="ignore requests logged before this ISO timestamp")
    args = ap.parse_args()
    from memobench import runtime  # noqa: F401  (loads .env; the client is built but never called)

    measured = [Path(d) / "llm" for d in args.measured]
    regen = tuple(args.regen) if args.regen else (0.5, 1.0)
    out = Path(args.out)
    result = dry_run(VILLAGES[args.village], measured, root=out / "llm", regen=regen,
                     verifier=MODELS[args.verifier] if args.verifier else None, since=args.since)  # fmt: skip
    (out / "dry_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'stage':28s} {'model':18s} {'requests':>12s} {'low $':>9s} {'high $':>9s}")
    for r in result["stages"]:
        print(f"{r['stage']:28s} {r['model']:18s} {str(r['requests']):>12s} {r['low_usd']:9.2f} {r['high_usd']:9.2f}")
    print(f"{'TOTAL':28s} {'':18s} {'':>12s} {result['total']['low_usd']:9.2f} {result['total']['high_usd']:9.2f}")
    print(f"buildable now: {result['buildable']}")


if __name__ == "__main__":
    main()
