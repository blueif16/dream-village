"""Per-stage cost/token/latency report from runs/<root>/logs/requests.jsonl (+ parse failures, cache hits).

Usage: uv run python scripts/ledger_report.py [run_id ...]   (default: all runs in runs/logs/requests.jsonl)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs"
runs = set(sys.argv[1:])
rows = [json.loads(l) for l in (ROOT / "logs/requests.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if not runs or r.get("run_id") in runs]

agg = defaultdict(lambda: defaultdict(float))
for r in rows:
    a = agg[(r.get("system") or "-", r.get("stage") or "-", r.get("mode"))]
    u = r.get("usage") or {}
    a["calls"] += 1
    a["in"] += u.get("input_tokens") or 0
    a["cache_w"] += u.get("cache_creation_input_tokens") or 0
    a["cache_r"] += u.get("cache_read_input_tokens") or 0
    a["out"] += u.get("output_tokens") or 0
    a["vis_chars"] += r.get("visible_text_chars") or 0
    a["cost"] += r.get("cost_usd") or 0
    a["lat"] += r.get("latency_ms") or 0
    a["max_tok_stops"] += r.get("stop_reason") == "max_tokens"

print(f"{'system':14}{'stage':13}{'mode':6}{'calls':>6}{'in/call':>9}{'cacheR%':>8}{'out/call':>9}{'think%':>7}{'$':>9}{'s/call':>7}{'cut':>4}")
tot = 0.0
for (sysn, stage, mode), a in sorted(agg.items()):
    n = a["calls"]
    prompt = a["in"] + a["cache_w"] + a["cache_r"]
    vis_tok = a["vis_chars"] / 3.5  # rough visible-token estimate; thinking = output - visible
    think = max(0.0, 1 - vis_tok / a["out"]) if a["out"] else 0
    print(f"{sysn:14}{stage:13}{mode or '-':6}{n:6.0f}{prompt / n:9.0f}{100 * a['cache_r'] / max(prompt, 1):8.1f}"
          f"{a['out'] / n:9.0f}{100 * think:7.0f}{a['cost']:9.4f}{a['lat'] / n / 1000:7.1f}{a['max_tok_stops']:4.0f}")
    tot += a["cost"]
print(f"total ${tot:.4f} over {len(rows)} requests")
pf = ROOT / "logs/parse_failures.jsonl"
print("parse failures:", len(pf.read_text().splitlines()) if pf.exists() else 0)
