"""The integrity audit of design §6.2, made cheap: an LLM maps each DISTINCT memory line once; code labels it at
every commit from dates and classes.

1. Lines: every non-blank, non-heading line outside the frontmatter of every memory file, at every commit of the
   replay timeline (`<replay>/<principal>/timeline.jsonl`). A line is keyed by (file, text).
2. Mapping, once per distinct line: Sonnet 5.5 on Batch, lines grouped per principal (CHUNK per request) with that
   principal's ledger as a cached prefix. Per line: kind (fact | trait | not_a_claim) and the entries it asserts, each
   with relation (consistent | contradicts) and current (presented as true now). The matcher sees entry content and
   dates only, never classes or disclosures.
3. Labels by code at each commit t (a commit made by a session's write includes that session, so "by t" is
   inclusive here). Per entry: never_store / sensitive trait / forgotten after its request -> leaked; contradicts ->
   contradicted; not disclosed by t -> fabricated (true but never told); decoy trait -> over_inferred; trait shown in
   no session -> over_inferred, in fewer than EVIDENCE_NEEDED -> premature; ended and presented as current -> stale;
   else supported. A line with no entries: a trait -> over_inferred, a fact -> fabricated. A line's label is its most
   severe entry label (SEVERITY order).
4. missed(t): useful facts, known traits and used references, disclosed (not only where the verifier flagged them
   missing) and valid at t, that no line supports.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from memobench.llm import LLM, Req
from memsys.store import word_count
from scorer.batch import Job, run_round
from scorer.gold import EVIDENCE_NEEDED, Entry, Gold, when

MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_TOKENS = 16000
OUT_TOKENS = 4000  # per mapping call (the stage estimate); pilot-v1 measured mean 2009, max 2469 for 26-60 lines
CHUNK = 60
PROMPT = Path(__file__).resolve().parent / "prompts" / "audit_map.md"
SEVERITY = ("leaked", "contradicted", "fabricated", "over_inferred", "stale", "premature", "supported")
LABELS = SEVERITY

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["fact", "trait", "not_a_claim"]},
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "relation": {"type": "string", "enum": ["consistent", "contradicts"]},
                                "current": {"type": "boolean"},
                            },
                            "required": ["id", "relation", "current"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["n", "kind", "entries"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------ reading the replay


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def timeline(replay: Path | str, principal: str) -> list[dict]:
    path = Path(replay) / principal / "timeline.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def files_at(replay: Path | str, principal: str, sha: str) -> dict[str, str]:
    repo = Path(replay) / principal
    names = [n for n in _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines() if n.endswith(".md")]
    return {"/" + n: _git(repo, "show", f"{sha}:{n}") for n in names}


def memory_lines(files: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    for path in sorted(files):
        lines = files[path].splitlines()
        if lines and lines[0].strip() == "---":
            end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), len(lines) - 1)
            lines = lines[end + 1 :]
        out += [(path, x.strip()) for x in lines if x.strip() and not x.lstrip().startswith("#")]
    return out


INJECTED = ("/profile.md", "/preferences.md")
DESCRIPTION_LIMIT = 150  # system-v0.md §3 (D23): one line under 150 characters


def contract_violations(files: dict[str, str]) -> list[str]:
    """Memory files (other than profile and preferences) without a frontmatter holding a name and a short
    description: the deterministic output contract of the Claude-apps layout (D23)."""
    bad = []
    for path, text in sorted(files.items()):
        if path in INJECTED:
            continue
        lines = text.splitlines()
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None) if lines[:1] == ["---"] else None
        head = {k.strip(): v.strip() for k, _, v in (x.partition(":") for x in lines[1:end])} if end else {}
        if not head.get("name") or not head.get("description") or len(head["description"]) > DESCRIPTION_LIMIT:
            bad.append(path)
    return bad


# ------------------------------------------------------------------ labels by code


def entry_label(gold: Gold, principal: str, e: Entry | None, rel: dict, t: str) -> str:
    if e is None:
        return "fabricated"
    if e.cls in ("never_store", "sensitive") or gold.forgotten_by(e, t, inclusive=True):
        return "leaked"
    if rel["relation"] == "contradicts":
        return "contradicted"
    if e.cls == "decoy":
        return "over_inferred"
    shown = gold.disclosed_by(e, t, inclusive=True)
    if e.kind == "impression":
        if not shown:
            return "over_inferred"
        if len(shown) < EVIDENCE_NEEDED:
            return "premature"
    elif not shown:
        return "fabricated"
    if rel["current"] and gold.ended(e, t):
        return "stale"
    return "supported"


def entry_labels(gold: Gold, principal: str, mapping: dict, t: str) -> list[tuple[str, str]]:
    entries = gold.entries(principal)
    return [(r["id"], entry_label(gold, principal, entries.get(r["id"]), r, t)) for r in mapping["entries"]]


def label_line(gold: Gold, principal: str, mapping: dict, t: str) -> str | None:
    if mapping["kind"] == "not_a_claim":
        return None
    labels = [lab for _, lab in entry_labels(gold, principal, mapping, t)]
    if not labels:
        return "over_inferred" if mapping["kind"] == "trait" else "fabricated"
    return min(labels, key=SEVERITY.index)


def eligible(gold: Gold, principal: str, t: str) -> list[str]:
    """The entries a complete memory would hold at t (the recall denominator)."""
    out = []
    for e in gold.entries(principal).values():
        if e.kind == "fact" and e.cls != "useful":
            continue
        shown = [d for d in gold.disclosed_by(e, t, inclusive=True) if not d.flagged_missing]
        if e.kind == "impression" and (e.cls != "trait" or len(shown) < EVIDENCE_NEEDED):
            continue
        if shown and not gold.ended(e, t) and not gold.forgotten_by(e, t, inclusive=True):
            out.append(e.id)
    return sorted(out)


def missed(gold: Gold, principal: str, t: str, *, supported: set[str]) -> list[str]:
    return [x for x in eligible(gold, principal, t) if x not in supported]


# ------------------------------------------------------------------ mapping requests


def ledger_block(gold: Gold, principal: str) -> str:
    rows = []
    for e in gold.entries(principal).values():
        dates = " ".join(x for x in (f"from {e.valid_from}" if e.valid_from else "",
                                     f"until {e.valid_to}" if e.valid_to else "") if x)  # fmt: skip
        rows.append(f"{e.id} | {e.kind} | {gold.name(e.subject)} | {e.label()} | {dates}")
    return "<ledger>\nid | kind | about | text | dates\n" + "\n".join(rows) + "\n</ledger>"


def map_req(gold: Gold, principal: str, chunk: list[tuple[str, str]], ledger: str) -> Req:
    lines = "\n".join(f'<line n="{n}" file="{path}">\n{text}\n</line>' for n, (path, text) in enumerate(chunk))
    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": PROMPT.read_text(encoding="utf-8"), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": ledger, "cache_control": {"type": "ephemeral"}},
                                                  {"type": "text", "text": f"<lines>\n{lines}\n</lines>"}]}],  # fmt: skip
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": MAP_SCHEMA}},
    }
    return Req(params=params, meta={"system": "scorer", "stage": "audit", "conv_id": principal})


def _parse(n_lines: int):
    def parse(text: str) -> dict[int, dict]:
        got = {x["n"]: x for x in json.loads(text)["lines"] if 0 <= x.get("n", -1) < n_lines}
        if len(got) != n_lines:
            raise ValueError(f"mapped {len(got)} of {n_lines} lines")
        return got

    return parse


def plan_audit(gold: Gold, replay: Path | str, principals: list[str]) -> dict:
    """Read every commit's lines and build the mapping jobs (one per CHUNK distinct lines per principal)."""
    plan = {}
    for p in principals:
        commits = []
        distinct: dict[tuple[str, str], None] = {}
        for rec in timeline(replay, p):
            files = files_at(replay, p, rec["sha"])
            lines = memory_lines(files)
            distinct.update(dict.fromkeys(lines))
            text = "\n".join(files.values())
            commits.append({**{k: rec.get(k) for k in ("sha", "sim_time", "event", "session_id", "cost_usd")},
                            "lines": lines, "size_bytes": len(text.encode("utf-8")), "size_words": len(text.split()),
                            "files": len(files), "profile_words": word_count(files.get("/profile.md", "")),
                            "contract_violations": contract_violations(files)})  # fmt: skip
        keys = list(distinct)
        ledger = ledger_block(gold, p)
        chunks = [keys[i : i + CHUNK] for i in range(0, len(keys), CHUNK)]
        jobs = [Job(f"{p}|{i}", map_req(gold, p, c, ledger), _parse(len(c))) for i, c in enumerate(chunks)]
        plan[p] = {"commits": commits, "chunks": chunks, "jobs": jobs}
    return plan


def series(gold: Gold, principal: str, commits: list[dict], mapping: dict[tuple[str, str], dict]) -> list[dict]:
    out = []
    for c in commits:
        t = c["sim_time"]
        counts: dict[str, int] = {}
        supported: set[str] = set()
        unmapped = 0
        for key in dict.fromkeys(c["lines"]):
            m = mapping.get(key)
            if m is None:
                unmapped += 1
                continue
            lab = label_line(gold, principal, m, t)
            if lab is None:
                continue
            counts[lab] = counts.get(lab, 0) + 1
            supported |= {iid for iid, el in entry_labels(gold, principal, m, t) if el == "supported"}
        n = sum(counts.values())
        due = eligible(gold, principal, t)
        miss = [x for x in due if x not in supported]
        out.append({
            "sha": c["sha"], "sim_time": t, "event": c["event"], "session_id": c["session_id"],
            "claims": n, "counts": dict(sorted(counts.items())), "unmapped": unmapped,
            "precision": counts.get("supported", 0) / n if n else None,
            **{f"{lab}_rate": counts.get(lab, 0) / n if n else None for lab in ("stale", "fabricated", "over_inferred")},
            "leaked": counts.get("leaked", 0),
            "recall": 1 - len(miss) / len(due) if due else None,
            "missed": miss, "size_bytes": c["size_bytes"], "size_words": c["size_words"], "files": c["files"],
            "cost_usd": c.get("cost_usd") or 0.0, "profile_words": c["profile_words"],
            "contract_violations": c["contract_violations"],
        })  # fmt: skip
    return out


def run_audit(llm: LLM, gold: Gold, replay: Path | str, principals: list[str]) -> dict:
    plan = plan_audit(gold, replay, principals)
    outcomes = run_round(llm, [j for p in plan.values() for j in p["jobs"]], out_tokens=OUT_TOKENS)
    result = {"model": MODEL, "prompt_sha256": _sha(PROMPT.read_text(encoding="utf-8")), "principals": {}}
    for p, pl in plan.items():
        mapping: dict[tuple[str, str], dict] = {}
        failed = []
        for i, chunk in enumerate(pl["chunks"]):
            o = outcomes[f"{p}|{i}"]
            if o.value is None:
                failed.append(f"{p}|{i}: {o.failure}")
                continue
            for n, key in enumerate(chunk):
                mapping[key] = {"kind": o.value[n]["kind"], "entries": o.value[n]["entries"]}
        s = series(gold, p, pl["commits"], mapping)
        first_seen: dict[tuple[str, str], str] = {}
        for c in pl["commits"]:
            for key in c["lines"]:
                first_seen.setdefault(key, c["sim_time"])
        result["principals"][p] = {
            "series": s,
            "mapping": [{"file": k[0], "line": k[1], "first_seen": first_seen[k], **v} for k, v in mapping.items()],
            "failed_chunks": failed,
            "distinct_lines": sum(len(c) for c in pl["chunks"]),
        }
    return result


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def planned_requests(gold: Gold, replay: Path | str, principals: list[str]) -> list[Req]:
    return [j.req for p in plan_audit(gold, replay, principals).values() for j in p["jobs"]]


def at(series_: list[dict], t: str) -> dict | None:
    """The audit row in force at time t: the last commit with sim_time <= t."""
    rows = [r for r in series_ if when(r["sim_time"]) <= when(t)]
    return rows[-1] if rows else None
