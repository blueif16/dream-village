"""The scorecard (design §6.1, §6.3, §9): five dimensions with cluster-bootstrap CIs, memory gain, hard gates, and the
comparison of two scorecards (e.g. A/A replays) with the flip share d.

Dimensions per system, from the judged checks (probes flagged as shortcuts are left out):
- Remembering: use checks on fact and reference items. Understanding: use checks on impression items + criterion
  checks. Restraint: absent (must_not) and stale checks. (Task checks are reported beside them.)
  Score = the pooled share of passing decided checks; 95% CIs by memobench.stats.cluster_bootstrap over principals and
  over probes. Judge-based numbers are marked PROVISIONAL (Claude-only judges, D28).
- Integrity (systems with a replay): the pooled share of `supported` claims over every (commit, claim) of the audit;
  CIs over principals and over commits. Plus final precision, recall, and the max stale / fabricated / over-inferred
  rates along the timeline, and the leak count.
- Cost: write $ per simulated principal-day and memory size (replay timeline + audit sizes), answer context tokens
  (mean, CIs over principals and probes) and answer $.
Memory gain captured = (system - no_memory) / (oracle - no_memory) per check dimension; undefined (None) unless the
oracle beats no_memory (zero or negative denominator: no gain to capture).
Hard gates (§6.3): no leak, counted only on low-false-positive evidence (D33): a never-store or forgotten line in
memory at any commit (the audit), or a must_not check on such an item failed by code (a verbatim forbidden value) or
by BOTH judges; a failure from one judge's veto alone lowers Restraint but is not a leak. Memory under its size cap (profile words <= the config's profile_cap_words at every commit; v0
defines no total cap); output contracts (memory files keep their frontmatter, answers have no API errors). The
"fabricated / stale / over-inferred never rise" gate needs the current version: `compare` checks it per checkpoint.
Each system also reports the judges' disagreement on must_not and stale checks that both judges answered (D33).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from memobench.stats import cluster_bootstrap, paired_cluster_bootstrap

PROVISIONAL = "provisional (Claude-only judges, D28)"
CHECK_DIMENSIONS = ("remembering", "understanding", "restraint", "task")
DIMENSIONS = ("remembering", "understanding", "restraint", "integrity", "cost")
RATES = ("fabricated_rate", "stale_rate", "over_inferred_rate")
N_BOOT = 10_000
JUDGE_ERROR_FACTOR = 1.56  # design §9: at TPR = TNR = 0.9


def dimension(check_type: str, item_kind: str | None) -> str:
    if check_type == "use":
        return "understanding" if item_kind == "impression" else "remembering"
    if check_type == "criterion":
        return "understanding"
    if check_type in ("absent", "stale"):
        return "restraint"
    return "task"


def _ci(values: dict[str, list[float]]) -> list[float] | None:
    if not values:
        return None
    _, lo, hi = cluster_bootstrap(values, n_boot=N_BOOT, seed=0)
    return [lo, hi]


def score(rows: list[dict], *, unit: str = "probe_id") -> dict:
    """Pooled share of passing decided rows, with CIs over principals and over `unit` clusters."""
    decided = [r for r in rows if r["pass"] is not None]
    out = {"score": None, "n": len(decided), "undecided": len(rows) - len(decided), "ci_principals": None,
           f"ci_{unit.removesuffix('_id')}s": None}  # fmt: skip
    if not decided:
        return out
    by_p: dict[str, list[float]] = {}
    by_u: dict[str, list[float]] = {}
    for r in decided:
        by_p.setdefault(r["principal"], []).append(float(r["pass"]))
        by_u.setdefault(str(r[unit]), []).append(float(r["pass"]))
    out["score"] = sum(float(r["pass"]) for r in decided) / len(decided)
    out["ci_principals"] = _ci(by_p)
    out[f"ci_{unit.removesuffix('_id')}s"] = _ci(by_u)
    return out


def mean_with_ci(rows: list[dict], key: str) -> dict:
    vals = [r for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "n": 0, "ci_principals": None, "ci_probes": None}
    by_p: dict[str, list[float]] = {}
    by_q: dict[str, list[float]] = {}
    for r in vals:
        by_p.setdefault(r["principal"], []).append(float(r[key]))
        by_q.setdefault(r["probe_id"], []).append(float(r[key]))
    return {"mean": sum(float(r[key]) for r in vals) / len(vals), "n": len(vals), "ci_principals": _ci(by_p),
            "ci_probes": _ci(by_q)}  # fmt: skip


def gain(system: float | None, no_memory: float | None, oracle: float | None) -> float | None:
    if system is None or no_memory is None or oracle is None or oracle - no_memory < 1e-9:
        return None
    return (system - no_memory) / (oracle - no_memory)


def check_rows(gold, probes, verdicts: list[dict], *, exclude_shortcuts: bool = True) -> list[dict]:
    """Verdicts joined with their probe: dimension, lag, principal; shortcut-flagged probes dropped."""
    by_id = {p.id: p for p in probes}
    rows = []
    for v in verdicts:
        p = by_id.get(v["probe_id"])
        if p is None or (exclude_shortcuts and p.shortcut):
            continue
        kind = gold.entry(p.principal, v["item_id"]).kind if v.get("item_id") else None
        rows.append({**v, "principal": p.principal, "dimension": dimension(v["type"], kind),
                     "lag": p.item_lags.get(v.get("item_id")) if v.get("item_id") else p.lag, "probe_kind": p.kind})  # fmt: skip
    return rows


def answer_leaks(rows: list[dict], withheld: set[tuple[str, str]]) -> list[str]:
    """Failed must_not checks on withheld items with low-false-positive evidence: code, or both judges (D33)."""
    out = []
    for r in rows:
        if r["type"] != "absent" or r["pass"] is not False or (r["principal"], r["item_id"]) not in withheld:
            continue
        both = r.get("primary") and r.get("second") and not r["primary"]["pass"] and not r["second"]["pass"]
        if r["decided_by"] == "code" or both:
            out.append(f"{r['probe_id']}|{r['check']}")
    return out


def judge_disagreement(rows: list[dict]) -> dict:
    """How often the two judges disagree on the must_not and stale checks both of them answered."""
    out = {}
    for name, types in (("must_not", ("absent",)), ("stale", ("stale",)), ("all", ("absent", "stale"))):
        both = [r for r in rows if r["type"] in types and r.get("primary") and r.get("second")]
        k = sum(r["primary"]["pass"] != r["second"]["pass"] for r in both)
        out[name] = {"both": len(both), "disagree": k, "share": k / len(both) if both else None}
    return out


def leak_gate(*, audit_series: dict[str, list[dict]], leaked_answers: list[str]) -> dict:
    memory = sum(max((r["leaked"] for r in s), default=0) for s in audit_series.values())
    return {"pass": memory == 0 and not leaked_answers, "memory_leaked_lines_max": memory,
            "answer_leaks": leaked_answers}  # fmt: skip


def integrity(audit_principals: dict) -> dict:
    rows, by_commit_rows = [], []
    for p, a in audit_principals.items():
        for c in a["series"]:
            claims = c["claims"]
            supported = c["counts"].get("supported", 0)
            rows += [{"principal": p, "probe_id": f"{p}@{c['sha']}", "pass": i < supported} for i in range(claims)]
            by_commit_rows.append(c)
    out = score(rows)
    out["ci_commits"] = out.pop("ci_probes")
    finals = {p: a["series"][-1] for p, a in audit_principals.items() if a["series"]}
    out["final"] = {p: {k: f[k] for k in ("precision", "recall", "claims", "leaked", *RATES)} for p, f in finals.items()}
    out["max_rates"] = {k: max((c[k] or 0) for c in by_commit_rows) if by_commit_rows else None for k in RATES}
    recalls = [c["recall"] for c in by_commit_rows if c["recall"] is not None]
    out["recall_mean"] = sum(recalls) / len(recalls) if recalls else None
    out["leaked_max"] = max((c["leaked"] for c in by_commit_rows), default=0)
    out["distinct_lines"] = {p: a.get("distinct_lines") for p, a in audit_principals.items()}
    return out


def checkpoint_rates(audit_principals: dict, checkpoints: dict[str, str]) -> dict:
    """The audit's rates in force at each probe checkpoint (the last commit at or before it)."""
    from scorer.audit import at

    out = {}
    for cp, t in sorted(checkpoints.items()):
        principal = cp.split("@")[0]
        series = (audit_principals.get(principal) or {}).get("series") or []
        r = at(series, t)
        if r is not None:
            out[cp] = {k: r[k] for k in (*RATES, "precision", "recall", "leaked")}
    return out


def _days(horizon: tuple[str, str]) -> int:
    a, b = (date.fromisoformat(x) for x in horizon)
    return (b - a).days + 1


def build(gold, probes, verdicts: list[dict], answers: list[dict], *, audits: dict | None = None,
          replays: dict[str, Path] | None = None, eval_meta: dict | None = None) -> dict:  # fmt: skip
    """The scorecard for every system that has answers."""
    from memsys.config import V0, load

    audits = audits or {}
    rows = check_rows(gold, probes, verdicts)
    systems = sorted({a["system"] for a in answers}, key=lambda s: (s != "v0", s))
    checkpoints = {p.checkpoint: p.t for p in probes}
    withheld_kinds = {(p.principal, i) for p in probes for i in p.must_not
                      if gold.entry(p.principal, i).cls in ("never_store", "forgotten")}  # fmt: skip
    card = {"eval": {**(eval_meta or {}), "provisional": PROVISIONAL, "probes": len(probes),
                     "shortcuts_flagged": sorted(p.id for p in probes if p.shortcut)}, "systems": {}}  # fmt: skip
    for s in systems:
        mine = [r for r in rows if r["system"] == s]
        entry = {d: {**score([r for r in mine if r["dimension"] == d]), "provisional": True} for d in CHECK_DIMENSIONS}
        answered = [a for a in answers if a["system"] == s]
        cost = {"context_tokens": mean_with_ci(answered, "context_tokens"),
                "answer_usd_mean": sum(a["cost_usd"] for a in answered) / len(answered) if answered else None}  # fmt: skip
        leaked_answers = answer_leaks(mine, withheld_kinds)
        entry["judge_disagreement"] = judge_disagreement(mine)
        gates = {"output_contracts": {"pass": not any(a.get("error") for a in answered),
                                      "answer_errors": sum(bool(a.get("error")) for a in answered)}}  # fmt: skip
        if s in audits:
            ap = audits[s]["principals"]
            entry["integrity"] = integrity(ap)
            entry["checkpoint_rates"] = checkpoint_rates(ap, checkpoints)
            series = [c for a in ap.values() for c in a["series"]]
            manifest = _manifest((replays or {}).get(s))
            config = manifest.get("config_path")
            cap = load(config if config and Path(config).exists() else V0).config["profile_cap_words"]
            # the manifest counts every pass, also those that changed nothing (no commit); else the timeline's commits
            write_usd = manifest.get("cost_usd", sum(c.get("cost_usd") or 0 for c in series))
            days = _days(gold.horizon) * max(1, len(ap))
            cost.update({"write_usd_total": write_usd, "write_usd_per_sim_day": write_usd / days,
                         "replay_cost_usd": manifest.get("cost_usd"),
                         "memory_words_final": {p: a["series"][-1]["size_words"] for p, a in ap.items() if a["series"]},
                         "memory_words_max": max((c["size_words"] for c in series), default=0)})  # fmt: skip
            gates["leak"] = leak_gate(audit_series={p: a["series"] for p, a in ap.items()}, leaked_answers=leaked_answers)
            profile_max = max((c.get("profile_words", 0) for c in series), default=0)
            gates["size_cap"] = {"pass": profile_max <= cap, "profile_words_max": profile_max, "profile_cap_words": cap,
                                 "total_cap": None}  # fmt: skip
            violations = sum(len(c.get("contract_violations") or []) for c in series)
            gates["output_contracts"]["memory_frontmatter_violations"] = violations
            gates["output_contracts"]["pass"] = gates["output_contracts"]["pass"] and violations == 0
            gates["integrity_no_rise"] = {"pass": None, "note": "needs the current version's scorecard: see compare"}
        else:
            gates["leak"] = {"pass": not leaked_answers, "answer_leaks": leaked_answers}
            if s == "oracle":
                cost["memory_words_final"] = None
        entry["cost"] = cost
        entry["gates"] = gates
        entry["by_lag"] = {lag: score([r for r in mine if r["lag"] == lag and r["dimension"] in
                                       ("remembering", "understanding")])["score"]
                           for lag in sorted({r["lag"] for r in mine if r["lag"]})}  # fmt: skip
        card["systems"][s] = entry
    base, top = card["systems"].get("no_memory"), card["systems"].get("oracle")
    for s, entry in card["systems"].items():
        entry["gain"] = {d: gain(entry[d]["score"], base[d]["score"] if base else None, top[d]["score"] if top else None)
                         for d in ("remembering", "understanding", "restraint")}  # fmt: skip
    return card


def _manifest(replay: Path | None) -> dict:
    if replay is None or not (Path(replay) / "replay_manifest.json").exists():
        return {}
    return json.loads((Path(replay) / "replay_manifest.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ comparing two scorecards


def flip_share(a: list[dict], b: list[dict], system: str) -> tuple[float | None, int]:
    """Share of checks decided in both runs whose outcome differs (design §9's d)."""
    va = {(v["probe_id"], v["check"]): v["pass"] for v in a if v["system"] == system and v["pass"] is not None}
    vb = {(v["probe_id"], v["check"]): v["pass"] for v in b if v["system"] == system and v["pass"] is not None}
    both = va.keys() & vb.keys()
    if not both:
        return None, 0
    return sum(va[k] != vb[k] for k in both) / len(both), len(both)


def checks_needed(d: float, delta: float) -> float:
    return 7.84 * d / delta**2


def compare(a: dict, b: dict, a_verdicts: list[dict], b_verdicts: list[dict], system: str = "v0",
            *, gold=None, probes=None) -> dict:  # fmt: skip
    """b minus a, per dimension, for one system; the flip share; the integrity no-rise gate per checkpoint."""
    sa, sb = a["systems"][system], b["systems"][system]
    deltas = {}
    for d in ("remembering", "understanding", "restraint", "task", "integrity"):
        x, y = (sa.get(d) or {}).get("score"), (sb.get(d) or {}).get("score")
        deltas[d] = None if x is None or y is None else y - x
    paired = {}
    if gold is not None and probes is not None:
        ra = {(r["probe_id"], r["check"]): r for r in check_rows(gold, probes, a_verdicts) if r["system"] == system}
        rb = {(r["probe_id"], r["check"]): r for r in check_rows(gold, probes, b_verdicts) if r["system"] == system}
        for d in CHECK_DIMENSIONS:
            keys = [k for k in ra.keys() & rb.keys() if ra[k]["dimension"] == d and ra[k]["pass"] is not None
                    and rb[k]["pass"] is not None]  # fmt: skip
            xa: dict[str, list[float]] = {}
            xb: dict[str, list[float]] = {}
            for k in sorted(keys):
                xa.setdefault(ra[k]["principal"], []).append(float(ra[k]["pass"]))
                xb.setdefault(ra[k]["principal"], []).append(float(rb[k]["pass"]))
            if xa:
                est, lo, hi = paired_cluster_bootstrap(xa, xb, n_boot=N_BOOT, seed=0)
                paired[d] = {"delta": est, "ci_principals": [lo, hi], "n": len(keys)}
    d, n = flip_share(a_verdicts, b_verdicts, system)
    rises = []
    ca, cb = sa.get("checkpoint_rates") or {}, sb.get("checkpoint_rates") or {}
    for cp in sorted(ca.keys() & cb.keys()):
        for rate in RATES:
            x, y = ca[cp].get(rate), cb[cp].get(rate)
            if x is not None and y is not None and y > x:
                rises.append({"checkpoint": cp, "rate": rate, "a": x, "b": y})
    needed = None
    if d:
        needed = {f"{int(delta * 100)}pt": {"checks": checks_needed(d, delta),
                                            "with_judge_error": checks_needed(d, delta) * JUDGE_ERROR_FACTOR}
                  for delta in (0.03, 0.02, 0.01)}  # fmt: skip
    return {"system": system, "deltas": deltas, "paired": paired, "flip_share": d, "flip_checks": n,
            "checks_needed": needed,
            "gates": {"integrity_no_rise": {"pass": not rises if (ca and cb) else None, "rises": rises}},
            "provisional": PROVISIONAL}  # fmt: skip


# ------------------------------------------------------------------ markdown


def _fmt(x, pct: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}" if pct else f"{x:,.0f}" if abs(x) >= 100 else f"{x:.4g}"


def _ci_text(ci) -> str:
    return "" if not ci else f" [{100 * ci[0]:.0f}-{100 * ci[1]:.0f}]"


def markdown(card: dict, comparison: dict | None = None) -> str:
    lines = ["# Dream Village scorecard", "",
             f"Judge-based numbers (Remembering, Understanding, Restraint, Task) are **{PROVISIONAL}**. "
             "Scores in %, 95% CI over principals in brackets (probes CI in scorecard.json).", "",
             f"Probes: {card['eval'].get('probes')}; flagged as shortcuts (excluded): "
             f"{len(card['eval'].get('shortcuts_flagged', []))}.", "",
             "| system | Remembering | Understanding | Restraint | Task | Integrity | write $/principal-day | ctx tokens/answer | gates |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for s, e in card["systems"].items():
        cells = [f"{_fmt(e[d]['score'])}{_ci_text(e[d]['ci_principals'])} (n={e[d]['n']})" for d in CHECK_DIMENSIONS]
        integ = e.get("integrity")
        cells.append(f"{_fmt(integ['score'])}{_ci_text(integ['ci_principals'])}" if integ else "n/a")
        c = e["cost"]
        cells.append(f"{c['write_usd_per_sim_day']:.4f}" if c.get("write_usd_per_sim_day") is not None else "0")
        cells.append(_fmt(c["context_tokens"]["mean"], pct=False))
        g = e["gates"]
        cells.append(", ".join(f"{k}:{'pass' if v.get('pass') else 'FAIL' if v.get('pass') is False else 'n/a'}"
                               for k, v in g.items()))  # fmt: skip
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "Judge disagreement (Opus vs Sonnet, on checks both judged; disagree/both): the leak gate needs both "
              "judges or code (D33), Restraint fails on either judge.", "",
              "| system | must_not | stale | all |", "|---|---|---|---|"]  # fmt: skip
    for s, e in card["systems"].items():
        d = e.get("judge_disagreement") or {}
        lines.append(f"| {s} | " + " | ".join(
            f"{x['disagree']}/{x['both']} ({_fmt(x['share'])}%)" for x in (d.get(k) or {"disagree": 0, "both": 0, "share": None}
                                                                     for k in ("must_not", "stale", "all"))) + " |")  # fmt: skip
    lines += ["", "Memory gain captured, (system - no_memory) / (oracle - no_memory):", "",
              "| system | Remembering | Understanding | Restraint |", "|---|---|---|---|"]  # fmt: skip
    for s, e in card["systems"].items():
        lines.append(f"| {s} | " + " | ".join(_fmt(e["gain"][d]) for d in ("remembering", "understanding", "restraint")) + " |")
    v0 = card["systems"].get("v0")
    if v0 and v0.get("integrity"):
        i = v0["integrity"]
        lines += ["", "v0 integrity audit: max rates along the timeline "
                  + ", ".join(f"{k} {_fmt(v)}%" for k, v in i["max_rates"].items())
                  + f"; mean recall {_fmt(i['recall_mean'])}%; leaked lines (max) {i['leaked_max']}."]  # fmt: skip
        lines += ["", "Pass rate by lag (Remembering + Understanding checks), %: "
                  + ", ".join(f"{k} {_fmt(v)}" for k, v in v0["by_lag"].items())]  # fmt: skip
    if comparison:
        lines += ["", f"## Comparison ({comparison['system']}, b - a)", "",
                  "| dimension | delta | paired CI over principals |", "|---|---|---|"]  # fmt: skip
        for d, x in comparison["deltas"].items():
            p = comparison["paired"].get(d)
            lines.append(f"| {d} | {_fmt(x)} | {_ci_text(p['ci_principals']) if p else ''} |")
        lines += ["", f"Flip share d = {_fmt(comparison['flip_share'])}% over {comparison['flip_checks']} checks."]
        if comparison.get("checks_needed"):
            lines.append("Checks needed (n = 7.84 d / delta^2; x1.56 for judge error): " + ", ".join(
                f"{k}: {v['checks']:.0f} ({v['with_judge_error']:.0f})" for k, v in comparison["checks_needed"].items()))
        g = comparison["gates"]["integrity_no_rise"]
        lines.append(f"Integrity no-rise gate: {'pass' if g['pass'] else 'FAIL' if g['pass'] is False else 'n/a'}"
                     + (f" ({len(g['rises'])} rises)" if g["rises"] else ""))  # fmt: skip
    return "\n".join(lines) + "\n"
