"""Export the Dream Village, its v0 replay and the scorer's outputs as a static JSON bundle for the viz app.

    .venv/bin/python viz/export.py [--village DIR] [--replay DIR] [--scorer DIR] [--out DIR]

Writes <out>/index.json (town, principals, gate, scorer status, scorecard summary, the optimization round-1 panel from
viz/round1.json and its candidates' scorecards) and <out>/p/<pid>.json (sessions,
memory commits with diff + full snapshot, ground truth, probes with answers and verdicts). Read-only on every input:
the replay repos are only read with `git ls-tree`, `git diff` and `git cat-file`. Re-run it as scorer files appear;
any missing scorer file is reported as absent, never filled in.

Never-store content is redacted in everything exported (visualization.md §9): never-store fact values, the verifier's
verbatim quotes of them, and the probes' forbidden strings become REDACTED. A minor principal's age is not exported.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the repo root, for scorer.gold when run as a script

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's well-known empty tree: the first commit's parent
REDACTED = "█" * 8  # a fixed-width bar: the length of the hidden text is not revealed
MIN_VALUE_CHARS = 6  # never-store values shorter than this ("16") are too generic to redact by string match
SCORER_FILES = ("probes.jsonl", "answers.jsonl", "verdicts.jsonl", "scorecard.json", "scorecard.md")


# ------------------------------------------------------------------ replay repos


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, check=True).stdout


def _blobs(repo: Path, shas: list[str]) -> dict[str, str]:
    """The text of each blob, read in one `git cat-file --batch` call."""
    out: dict[str, str] = {}
    if not shas:
        return out
    raw = _git(repo, "cat-file", "--batch", stdin="".join(s + "\n" for s in shas).encode())
    i = 0
    for sha in shas:
        nl = raw.index(b"\n", i)
        size = int(raw[i:nl].split()[2])
        out[sha] = raw[nl + 1 : nl + 1 + size].decode("utf-8", errors="replace")
        i = nl + 1 + size + 1  # the content is followed by a newline
    return out


def read_commits(repo: Path) -> list[dict]:
    """Every commit in the replay's timeline.jsonl, in order, with the files it changed (status A/M/D), its unified
    diff against the previous commit (the empty tree for the first) and the full memory snapshot {path: content}."""
    repo = Path(repo)
    timeline = [json.loads(x) for x in (repo / "timeline.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    trees: list[dict[str, str]] = []  # per commit: path -> blob sha
    for rec in timeline:
        tree = {}
        for line in _git(repo, "ls-tree", "-r", rec["sha"]).decode().splitlines():
            meta, path = line.split("\t", 1)
            _mode, kind, blob = meta.split()
            if kind == "blob":
                tree[path] = blob
        trees.append(tree)
    texts = _blobs(repo, sorted({b for t in trees for b in t.values()}))
    commits = []
    prev = EMPTY_TREE
    for rec, tree in zip(timeline, trees):
        sha = rec["sha"]
        status = _git(repo, "diff", "--no-renames", "--name-status", prev, sha).decode().splitlines()
        files = [{"path": p, "status": s[0]} for s, p in (x.split("\t", 1) for x in status if x)]
        diff = _git(repo, "diff", "--no-renames", "--no-color", prev, sha).decode("utf-8", errors="replace")
        commits.append({
            "sha": sha, "parent": None if prev == EMPTY_TREE else prev, "sim_time": rec.get("sim_time"),
            "event": rec.get("event"), "session_id": rec.get("session_id"), "cost_usd": rec.get("cost_usd"),
            "tokens": rec.get("tokens"), "files": files, "diff": diff,
            "snapshot": {path: texts[blob] for path, blob in sorted(tree.items())},
        })  # fmt: skip
        prev = sha
    return commits


def memory_lines(snapshot: dict[str, str]) -> int:
    """Lines of memory content: non-blank, not a heading, outside YAML frontmatter (as scorer.audit counts them)."""
    n = 0
    for text in snapshot.values():
        lines = text.splitlines()
        if lines and lines[0].strip() == "---":
            end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), len(lines) - 1)
            lines = lines[end + 1 :]
        n += sum(1 for x in lines if x.strip() and not x.lstrip().startswith("#"))
    return n


# ------------------------------------------------------------------ redaction


class Redactor:
    def __init__(self, needles: set[str]) -> None:
        needles = {n.strip() for n in needles if n and len(n.strip()) >= MIN_VALUE_CHARS}
        alts = sorted(needles, key=len, reverse=True)  # longest first, so a quote wins over a value inside it
        self.rx = re.compile("|".join(re.escape(n) for n in alts), re.IGNORECASE) if alts else None
        self.hits = 0

    def __call__(self, text: str | None) -> str | None:
        if not text or self.rx is None:
            return text
        out, n = self.rx.subn(REDACTED, text)
        self.hits += n
        return out


# ------------------------------------------------------------------ helpers


def _read_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a file still being written: keep the complete records before the torn line
    return rows


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


def _entry(e, redact: Redactor) -> dict:
    never = e.cls == "never_store"
    return {
        "id": e.id, "kind": e.kind, "subject": e.subject, "attribute": e.attribute, "cls": e.cls,
        "text": REDACTED if never else redact(e.text), "valid_from": e.valid_from, "valid_to": e.valid_to,
        "superseded_by": e.superseded_by, "disclosed_in": [d.session_id for d in e.disclosures],
        "forget_at": e.forget_at,
    }  # fmt: skip


def _scorecard_summary(card: dict | None) -> dict | None:
    if not card:
        return None
    dims = ("remembering", "understanding", "restraint", "task")
    systems = {}
    for name, s in card.get("systems", {}).items():
        row = {d: {k: (s.get(d) or {}).get(k) for k in ("score", "n", "ci_probes")} for d in dims}
        integ = s.get("integrity")
        row["integrity"] = {"score": integ.get("score"), "n": integ.get("n")} if integ else None
        row["gain"] = s.get("gain")
        row["gates"] = {g: v.get("pass") for g, v in (s.get("gates") or {}).items()}
        cost = s.get("cost") or {}
        row["cost"] = {"answer_usd_mean": cost.get("answer_usd_mean"),
                       "context_tokens_mean": (cost.get("context_tokens") or {}).get("mean"),
                       "write_usd_total": cost.get("write_usd_total")}  # fmt: skip
        systems[name] = row
    ev = card.get("eval", {})
    return {"probes": ev.get("probes"), "provisional": ev.get("provisional"), "rubric_version": ev.get("rubric_version"),
            "systems": systems}  # fmt: skip


# ------------------------------------------------------------------ optimization round 1

PAIRED_DIMS = ("remembering", "understanding", "restraint")  # the scorer pairs check dimensions only, never integrity


def _points(x: float | None) -> float | None:
    """A fraction as points: x100, one decimal (never -0.0)."""
    return None if x is None else round(x * 100, 1) + 0.0


def _comparison_run(root: Path, rel: str) -> dict:
    """One candidate scorecard's comparison with v0, in points. A check dimension is its `paired` delta (on the checks
    both runs decided) with that pair's CI over principals; integrity is the scorer's unpaired delta, with no CI. A
    missing scorecard is absent and never filled in."""
    card = _read_json(root / rel / "scorecard.json")
    out = {"scorecard": rel, "present": card is not None, "probes": None, "principals": None, "provisional": None,
           "dims": None}  # fmt: skip
    if card is None:
        return out
    cmp = card.get("comparison")
    integ = (card.get("systems", {}).get((cmp or {}).get("system", "v0")) or {}).get("integrity") or {}
    out["probes"] = card.get("eval", {}).get("probes")
    out["principals"] = sorted(integ.get("final") or {}) or None
    if not cmp:
        return out
    out["provisional"] = cmp.get("provisional")
    paired = cmp.get("paired") or {}
    dims: dict[str, dict | None] = {}
    for d in PAIRED_DIMS:
        p = paired.get(d)
        dims[d] = None if p is None else {"delta": _points(p.get("delta")), "source": "paired",
                                          "ci": [_points(x) for x in p["ci_principals"]] if p.get("ci_principals") else None}  # fmt: skip
    integ_delta = (cmp.get("deltas") or {}).get("integrity")
    dims["integrity"] = None if integ_delta is None else {"delta": _points(integ_delta), "ci": None, "source": "deltas"}
    out["dims"] = dims
    return out


def round1_summary(spec: dict, root: Path) -> dict:
    """The optimization round-1 panel: the hand-authored lesson, rule and verdicts from `spec` (viz/round1.json), with
    each candidate's screen and confirm comparison and the A/A noise floor read from scorecards whose repo-relative
    paths resolve against `root` (the checkout that holds runs/)."""
    nf = spec["noise_floor"]
    return {
        "lesson": spec["lesson"], "rule": spec["rule"],
        "noise_floor": {"label": nf.get("label"), **_comparison_run(root, nf["scorecard"])},
        "candidates": [{**{k: c.get(k) for k in ("id", "change", "verdict", "reason")},
                        "screen": _comparison_run(root, c["screen"]) if c.get("screen") else None,
                        "confirm": _comparison_run(root, c["confirm"]) if c.get("confirm") else None}
                       for c in spec["candidates"]],
    }  # fmt: skip


# ------------------------------------------------------------------ town: places, homes, where each principal is each day

# The short names full13's life logs use for its places. Every place also matches its full name and the part before a
# comma. Matching ignores case and accents and needs whole words.
ALIASES = {
    "pl-flats-high": ["Flats High", "high school"],
    "pl-tamsen-college": ["TVC", "Tamsen Valley College"],
    "pl-bayshore-state": ["Bayshore State", "Bayshore"],
    "pl-st-anselm": ["St. Anselm", "St Anselm", "4 West", "the hospital"],
    "pl-ridgeline": ["Ridgeline", "RF-4", "the warehouse"],
    "pl-marisol-apts": ["Marisol Gardens"],
    "pl-verdant-commons": ["Verdant Commons"],
    "pl-ruiz-market": ["Ruiz Family Market", "Ruiz"],
    "pl-flats-junction": ["Flats Junction", "Tule Line"],
    "pl-taqueria-colima": ["Taqueria Colima", "Colima"],
    "pl-pho-saigon-bay": ["Pho Ben Thanh"],
    "pl-cafe-heron": ["Heron & Grind", "Heron and Grind", "Heron"],
    "pl-boba-cloud": ["Cloud Nine", "boba"],
    "pl-plaza-marisol": ["Plaza Marisol"],
    "pl-ironflats-gym": ["Iron Flats", "the gym"],
    "pl-salt-marsh-park": ["Ulloa", "salt marsh", "marsh trail", "boardwalk"],
    "pl-central-park": ["Delgado Park", "Delgado"],
    "pl-flats-library": ["library", "homework center"],
    "pl-our-lady": ["Our Lady", "parish", "church"],
    "pl-gurdwara": ["gurdwara", "langar"],
    "pl-community-center": ["community center", "rec center"],
    "pl-suds-laundromat": ["Suds City", "laundromat"],
    "pl-panaderia": ["Panaderia", "Las Hermanas"],
    "pl-flats-middle": ["Kessler"],
    "pl-ruel-road": ["Ruel Road"],
    "pl-ortega-theater": ["Ortega Playhouse", "playhouse"],
}
SHARES_HOME = ("mother", "father", "parent", "daughter", "son", "roommate", "neighbor")  # a principal without a home fact


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def place_matcher(places: list[dict]):
    """text -> the id of the place it names first (the longer name wins a tie), or None."""
    pats = []
    for p in places:
        for name in {p["name"], p["name"].split(",")[0], *ALIASES.get(p["id"], [])}:
            pats.append((re.compile(r"(?<!\w)" + re.escape(_fold(name)) + r"(?!\w)"), p["id"], len(name)))

    def match(text: str) -> str | None:
        t = _fold(text)
        best = None
        for rx, pid, n in pats:
            m = rx.search(t)
            if m and (best is None or (m.start(), -n) < best[0]):
                best = ((m.start(), -n), pid)
        return best[1] if best else None

    return match


def day_positions(entries: list[dict], days: list[str], places: list[dict], home: str | dict | None,
                  anchor: str | None) -> dict[str, dict]:  # fmt: skip
    """Where a principal is on each day: the first place that day's life-log entries name; else home (a place id, or
    {day: place id} when it moves); else their school or work (`anchor`). Each day keeps its life-log lines."""
    match = place_matcher(places)
    by_day: dict[str, list[dict]] = {}
    for e in entries:
        by_day.setdefault(e["date"], []).append(e)
    out = {}
    for d in days:
        lines = by_day.get(d, [])
        place, why = next(((p, "life_log") for p in (match(e["what"]) for e in lines) if p), (None, None))
        if place is None:
            h = home.get(d) if isinstance(home, dict) else home
            place, why = (h, "home") if h else (anchor, "work") if anchor else (None, None)
        out[d] = {"place": place, "why": why, "lines": lines}
    return out


def _district(kind: str) -> str:
    if kind in ("high_school", "middle_school", "community_college", "state_university", "library"):
        return "learning"
    if kind in ("hospital", "community_center", "church", "temple", "theater"):
        return "civic"
    if kind in ("warehouse_employer", "industrial_area"):
        return "industry"
    if kind in ("apartment_complex", "house"):
        return "homes"
    if kind == "park":
        return "parks"
    if kind == "transit_station":
        return "station"
    return "main_street"


def town_bundle(village: Path, gold, people: dict, world: dict, inputs: dict, redact: Redactor) -> dict:
    places = [{"id": p["id"], "name": p["name"], "kind": p["kind"], "description": redact(p["description"]),
               "district": _district(p["kind"])} for p in world["places"]]  # fmt: skip
    match = place_matcher(places)
    starts = sorted(s["start"] for i in inputs.values() for s in i["sessions"])
    d0, d1 = date.fromisoformat(starts[0][:10]), date.fromisoformat(starts[-1][:10])
    days = [(d0 + timedelta(n)).isoformat() for n in range((d1 - d0).days + 1)]

    # homes: a home fact naming a place (an apartment complex) lives there; any other becomes a house of its own
    homes: dict[str, dict[str, str]] = {}  # principal -> {day: place id}
    for pid in gold.principals:
        facts = [e for e in gold.entries(pid).values() if e.attribute == "home" and e.subject == pid
                 and e.cls != "never_store"]  # fmt: skip
        for e in facts:
            place = match(e.text)
            if place is None:
                place = f"home-{e.id}"
                first = people[pid]["name"].split()[0]
                places.append({"id": place, "name": f"{first}'s home", "kind": "house", "description": redact(e.text),
                               "district": "homes"})  # fmt: skip
            for d in days:
                if (e.valid_from or "") <= d and (e.valid_to is None or d < e.valid_to):
                    homes.setdefault(pid, {})[d] = place
    cast = json.loads((village / "cast.json").read_text(encoding="utf-8"))
    for pid in gold.principals:
        if pid not in homes:
            other = next((ln["b"] for ln in cast["links"] if ln["a"] == pid and ln["b"] in homes
                          and any(w in ln["relation"] for w in SHARES_HOME)), None)  # fmt: skip
            if other is None:
                other = next((ln["a"] for ln in cast["links"] if ln["b"] == pid and ln["a"] in homes
                              and any(w in ln["relation"] for w in SHARES_HOME)), None)  # fmt: skip
            if other:
                homes[pid] = dict(homes[other])
            else:
                place = f"home-{pid}"
                places.append({"id": place, "name": f"{people[pid]['name'].split()[0]}'s home", "kind": "house",
                               "description": "", "district": "homes"})  # fmt: skip
                homes[pid] = {d: place for d in days}

    names = {p["id"]: p["name"] for p in cast["people"]}
    rows = []
    for pid in gold.principals:
        entries = []
        for f in sorted((village / "life_logs").glob(f"{pid}.w*.json"), key=lambda f: int(f.stem.split(".w")[1])):
            entries += json.loads(f.read_text(encoding="utf-8"))["entries"]
        anchor = match(people[pid]["role"])
        pos = day_positions(entries, days, places, homes.get(pid), anchor)
        sessions_by_day: dict[str, list[str]] = {}
        for s in inputs[pid]["sessions"]:
            sessions_by_day.setdefault(s["start"][:10], []).append(s["id"])
        minor = people[pid]["age"] < 18
        rows.append({
            "id": pid, "name": people[pid]["name"], "age": None if minor else people[pid]["age"], "minor": minor,
            "role": people[pid]["role"], "anchor": anchor, "homes": sorted(set(homes.get(pid, {}).values())),
            "sessions": sessions_by_day, "memory": inputs[pid]["memory"],
            "days": [{"place": pos[d]["place"], "why": pos[d]["why"],
                      "lines": [{"what": redact(e["what"]), "with": [names.get(w, w) for w in e.get("with", [])]}
                                for e in pos[d]["lines"]]} for d in days],
        })  # fmt: skip
    events = [{k: e.get(k) for k in ("id", "date", "end_date", "title", "description", "places")} for e in world["events"]]
    return {"town": world["town"], "days": days, "places": places, "events": events, "principals": rows}


# ------------------------------------------------------------------ export


def _public_path(path: Path) -> str:
    """A source path safe to publish: relative to the repo root, else just its last part (never a local absolute path)."""
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(HERE.parent.resolve()).as_posix()
    except ValueError:
        return path.name


def export(village: Path, replay: Path, scorer: Path, out: Path, round1: Path | None = None) -> dict:
    from scorer.gold import Gold

    gold = Gold(village)
    cast = json.loads((village / "cast.json").read_text(encoding="utf-8"))
    world = json.loads((village / "world.json").read_text(encoding="utf-8"))
    manifest = json.loads((village / "manifest.json").read_text(encoding="utf-8"))
    plans = {p["session_id"]: p for p in json.loads((village / "plans.json").read_text(encoding="utf-8"))}
    people = {p["id"]: p for p in cast["people"]}

    probes = _read_jsonl(scorer / "probes.jsonl")
    answers = _read_jsonl(scorer / "answers.jsonl")
    verdicts = _read_jsonl(scorer / "verdicts.jsonl")
    card = _read_json(scorer / "scorecard.json")
    scorer_status = {
        "dir": _public_path(scorer), "files": {f: (scorer / f).exists() for f in SCORER_FILES},
        "probes": len(probes) if probes is not None else None,
        "answers": None if answers is None else {s: sum(a["system"] == s for a in answers)
                                                 for s in sorted({a["system"] for a in answers})},
        "verdicts": len(verdicts) if verdicts is not None else None,
    }  # fmt: skip

    # never-store needles, village-wide: a principal may mention someone else's never-store fact
    never = {e.id: e.text for p in gold.principals for e in gold.entries(p).values() if e.cls == "never_store"}
    needles = set(never.values())
    needles |= {tok for v in never.values() for tok in re.findall(r"\b[A-Za-z]?\d[\d-]{4,}\d\b", v)}  # bare ID numbers
    for f in sorted((village / "verify").glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")).get("sensitive", []):
            needles |= {s.get("user_quote", ""), s.get("assistant_quote", "")}
    for pr in probes or []:  # forbidden also lists forgotten facts, which are shown; only never-store ones are hidden
        needles |= {v for item, vals in (pr.get("forbidden") or {}).items() if item in never for v in vals}
    redact = Redactor(needles)

    (out / "p").mkdir(parents=True, exist_ok=True)
    starts = []  # every session start and commit time: the span the timeline axis covers
    summaries = []
    town_inputs: dict[str, dict] = {}
    for pid in gold.principals:
        person = people[pid]
        bible = gold.bibles[pid]
        entries = gold.entries(pid)
        minor = person["age"] < 18
        sessions = []
        for s in gold.sessions(pid):
            plan = plans.get(s.id, {})
            v = _read_json(village / "verify" / f"{s.id}.json")
            items = []
            for it in plan.get("items", []):
                e = entries.get(it["item_id"])
                never = it.get("never_store") or (e is not None and e.cls == "never_store")
                items.append({
                    "item_id": it["item_id"], "mode": it["mode"], "never_store": bool(it.get("never_store")),
                    "forget_request": bool(it.get("forget_request")), "note": REDACTED if never else redact(it["note"]),
                    "truth": None if e is None else {"kind": e.kind, "attribute": e.attribute, "cls": e.cls,
                                                     "text": REDACTED if never else redact(e.text)},
                })  # fmt: skip
            sessions.append({
                "id": s.id, "start": s.start, "task": redact(plan.get("task")), "task_type": plan.get("task_type"),
                "turns": [{"role": t.role, "text": redact(t.text)} for t in s.turns], "items": items,
                "verify": None if v is None else {
                    "passed": v.get("passed"), "attempt": v.get("attempt"),
                    "flags": [{**fl, "detail": redact(fl.get("detail"))} for fl in v.get("flags", [])]},
            })  # fmt: skip
            starts.append(s.start)

        repo = replay / pid
        commits = read_commits(repo) if (repo / "timeline.jsonl").exists() else []
        for c in commits:
            starts.append(c["sim_time"])
            c["diff"] = redact(c["diff"])
            c["snapshot"] = {p: redact(t) for p, t in c["snapshot"].items()}
            c["lines"] = memory_lines(c["snapshot"])

        mine = [p for p in probes or [] if p["principal"] == pid]
        probe_rows = []
        for pr in mine:
            ans = {a["system"]: {"text": redact(a.get("text")), "refused": a.get("refused"), "error": a.get("error"),
                                 "context_tokens": a.get("context_tokens"), "tool_calls": a.get("tool_calls")}
                   for a in answers or [] if a["probe_id"] == pr["id"]}  # fmt: skip
            ver: dict[str, list] = {}
            for vd in verdicts or []:
                if vd["probe_id"] == pr["id"]:
                    judged = {k: ({**vd[k], "evidence": redact(vd[k].get("evidence"))} if vd.get(k) else None)
                              for k in ("primary", "second")}  # fmt: skip
                    ver.setdefault(vd["system"], []).append({
                        "check": vd.get("check"), "type": vd.get("type"), "item_id": vd.get("item_id"),
                        "pass": vd.get("pass"), "decided_by": vd.get("decided_by"), "code_rule": redact(vd.get("code_rule")),
                        **judged})  # fmt: skip
            probe_rows.append({
                "id": pr["id"], "checkpoint": pr["checkpoint"], "t": pr["t"], "kind": pr["kind"], "lag": pr.get("lag"),
                "request": redact(pr["request"]), "plan": redact(pr.get("plan")),
                "must_use": pr.get("must_use", []), "may_use": pr.get("may_use", []),
                "must_not": pr.get("must_not", []), "stale": pr.get("stale", []), "criteria": pr.get("criteria", []),
                "forbidden": sorted((pr.get("forbidden") or {}).keys()), "item_lags": pr.get("item_lags", {}),
                "shortcut": pr.get("shortcut"), "answers": ans, "verdicts": ver if verdicts is not None else None,
            })  # fmt: skip

        town_inputs[pid] = {"sessions": [{"id": s["id"], "start": s["start"]} for s in sessions],
                            "memory": [{"t": c["sim_time"], "lines": c["lines"], "files": len(c["snapshot"])}
                                       for c in commits]}  # fmt: skip
        truth = [_entry(e, redact) for e in entries.values()]
        bundle = {
            "principal": {"id": pid, "name": person["name"], "age": None if minor else person["age"], "minor": minor,
                          "role": person["role"], "usage": {"frequency": bible.usage.frequency,
                                                            "active_hours": bible.usage.active_hours}},
            "sessions": sessions, "commits": commits, "truth": truth, "probes": probe_rows,
        }  # fmt: skip
        path = out / "p" / f"{pid}.json"
        path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        last = commits[-1] if commits else None
        summaries.append({
            **bundle["principal"], "sessions": len(sessions), "commits": len(commits),
            "writes": sum(c["event"] == "write" for c in commits), "dreams": sum(c["event"] == "dream" for c in commits),
            "memory_lines": last["lines"] if last else 0, "memory_files": len(last["snapshot"]) if last else 0,
            "verify_passed": sum(bool(s["verify"] and s["verify"]["passed"]) for s in sessions),
            "probes": len(probe_rows), "replay_usd": sum(c["cost_usd"] or 0 for c in commits),
            "bundle_bytes": path.stat().st_size,
        })  # fmt: skip
        print(f"[viz] {pid}: {len(sessions)} sessions, {len(commits)} commits, {len(probe_rows)} probes, "
              f"{path.stat().st_size / 1e6:.2f} MB")  # fmt: skip

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "village": manifest.get("village"),
        "town": {"name": world["town"], "region": world["region"], "summary": world["summary"],
                 "places": len(world["places"]), "events": [{k: e[k] for k in ("date", "end_date", "title")}
                                                            for e in world["events"]]},  # fmt: skip
        "range": {"start": min(starts)[:10], "end": max(starts)[:10]},
        "principals": summaries,
        "gate": manifest.get("gate"),
        "scorer": scorer_status,
        "scorecard": _scorecard_summary(card),
        "redactions": redact.hits,
        "sources": {"village": _public_path(village), "replay": _public_path(replay), "scorer": _public_path(scorer)},
    }  # fmt: skip
    spec = _read_json(round1) if round1 else None
    index["round1"] = round1_summary(spec, main_root()) if spec else None
    town = town_bundle(village, gold, people, world, town_inputs, redact)
    (out / "town.json").write_text(json.dumps(town, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[viz] wrote {out / 'index.json'}; scorer: {scorer_status['files']}; {redact.hits} redactions")
    return index


def main_root() -> Path:
    """The main checkout (also from a worktree), where data/ and runs/ live: the parent of git's common dir."""
    try:
        common = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        return Path(common).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        return HERE.parent


def main(argv: list[str] | None = None) -> int:
    root = main_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--village", type=Path, default=root / "data/villages/full13-v1")
    ap.add_argument("--replay", type=Path, default=root / "runs/replays/full13-v1-v0")
    ap.add_argument("--scorer", type=Path, default=root / "runs/scorer/full13-v1-v0")
    ap.add_argument("--out", type=Path, default=HERE / "public/data")
    ap.add_argument("--round1", type=Path, default=HERE / "round1.json", help="the round-1 panel's hand-authored part")
    a = ap.parse_args(argv)
    export(a.village, a.replay, a.scorer, a.out, a.round1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

