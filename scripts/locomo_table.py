"""LoCoMo results table: LLM-judge accuracy with 95% CIs clustered by conversation (all-10 and held-out 6-10),
paired Memora-S vs -P, $/conversation (build vs query, judge excluded), prompt tokens and LLM calls per query.

Usage: uv run python scripts/locomo_table.py system=path/to/<method>_eval.json [system=...] [--query-run locomo-full]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from memobench.stats import cluster_bootstrap, paired_cluster_bootstrap

REPO = Path(__file__).resolve().parents[1]
PAPER = {"memora-p": 0.863, "memora-s": 0.849, "full-context": 0.825, "memora-noabs": 0.653, "rag": 0.633}
BUILD_RUNS = {"memora-s": ["locomo-build", "calib-c1"], "memora-noabs": ["locomo-build"]}
STORE_OF = {"memora-p": "memora-s", "memora-s": "memora-s", "memora-noabs": "memora-noabs"}

args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
query_run = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--query-run=")), "locomo-full")
order = [c["sample_id"] for c in json.loads((REPO / "data/locomo10.json").read_text())]


def by_name(ev):
    """Memora results key conversations by index ("0".."9"); baselines by sample_id. Normalize to sample_id."""
    return {order[int(c)] if c.isdigit() else c: items for c, items in ev.items()}


evals = {k: by_name(json.loads(Path(v).read_text())) for k, v in (a.split("=", 1) for a in args)}
held_out = set(order[5:])
logs = [json.loads(l) for l in (REPO / "runs/logs/requests.jsonl").read_text().splitlines() if l.strip()]


def scores(ev, convs=None):
    return {c: [float(it["llm_score"]) for it in items] for c, items in ev.items() if convs is None or c in convs}


def cost(system, runs, stages_excluded=("judge",)):
    return sum(r["cost_usd"] for r in logs if r.get("run_id") in runs and r.get("system") == system
               and r.get("stage") not in stages_excluded)


print(f"{'system':14}{'all-10 (95% CI)':>24}{'held-out 6-10':>24}{'paper':>7}{'n':>6}"
      f"{'$build/conv':>12}{'$query/conv':>12}{'prompt tok/q':>13}{'calls/q':>8}")
for name, ev in evals.items():
    est, lo, hi = cluster_bootstrap(scores(ev))
    hest, hlo, hhi = cluster_bootstrap(scores(ev, held_out))
    n = sum(len(v) for v in ev.values())
    q_rows = [r for r in logs if r.get("run_id") == query_run and r.get("system") == name and r.get("stage") != "judge"]
    calls_q = len(q_rows) / max(n, 1)
    ans = [r for r in q_rows if r.get("stage") == "answer"]
    tok_q = sum(sum((r["usage"].get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                for r in ans) / max(len(ans), 1)
    b = cost(STORE_OF[name], BUILD_RUNS[STORE_OF[name]]) / len(ev) if name in STORE_OF else 0.0
    q = sum(r["cost_usd"] for r in q_rows) / len(ev)
    print(f"{name:14}{est:8.3f} [{lo:.3f},{hi:.3f}]{hest:9.3f} [{hlo:.3f},{hhi:.3f}]{PAPER.get(name, float('nan')):7.3f}{n:6d}"
          f"{b:12.4f}{q:12.4f}{tok_q:13.0f}{calls_q:8.2f}")

    cats = defaultdict(list)
    for items in ev.values():
        for it in items:
            cats[it["category"]].append(float(it["llm_score"]))
    print(" " * 14 + "  per category: " + "  ".join(f"c{c}={sum(v) / len(v):.3f} (n={len(v)})" for c, v in sorted(cats.items())))

if {"memora-s", "memora-p"} <= evals.keys():
    def keyed(ev):  # (question, category, occurrence): 11 (question, category) pairs repeat within conversations
        out = {}
        for c, items in ev.items():
            seen = defaultdict(int)
            out[c] = {}
            for it in items:
                k = (it["question"], it["category"])
                out[c][k + (seen[k],)] = float(it["llm_score"])
                seen[k] += 1
        return out
    s, p = keyed(evals["memora-s"]), keyed(evals["memora-p"])
    common = {c: sorted(set(s[c]) & set(p[c])) for c in s if c in p}
    a = {c: [s[c][k] for k in ks] for c, ks in common.items()}
    b = {c: [p[c][k] for k in ks] for c, ks in common.items()}
    d, lo, hi = paired_cluster_bootstrap(a, b)
    print(f"\nPaired P - S on {sum(len(v) for v in a.values())} shared questions: {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
judge = sum(r["cost_usd"] for r in logs if r.get("run_id") == query_run and r.get("stage") == "judge")
print(f"judge cost (all systems, excluded above): ${judge:.4f}")
