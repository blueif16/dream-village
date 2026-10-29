"""Replay a village's sessions through system v0 and record every memory state in git (design.md §5).

Per principal, in time order: a write pass after each session; a dream whenever the simulated clock crosses the
cadence (v0: the first 03:00 after activity), including a trailing one after the last session unless --until cuts
it off. Principals run in parallel threads; each chain is sequential and real-time (LLM.call).

Layout of <out>/:
  replay_manifest.json         config + prompt hashes, models, counts, cost, tokens, wall time
  <principal>/                 a git repo holding only the memory files (one commit per state change)
  <principal>/timeline.jsonl   one record per commit: sha, sim_time, event, session_id, config_sha256, models,
                               cost_usd, tokens (git-excluded sidecar)
  <principal>/events.jsonl     one record per pass, committed or not (refusals, errors, no-op passes)

Commits carry the sim time as author and committer date (naive local sim time written as +00:00). A write commit's
sim_time is its session's start (sessions carry no end time). `checkout(out, principal, t)` is a read-only fork: a
fresh temp dir with the memory at the last commit whose sim_time <= t (empty before the first commit).

CLI:
  python -m memsys.replay --sessions <path> --out <dir> --config <file> [--principals a,b] [--until <date>]
                          [--salt <s>] [--budget <usd>]
--budget caps this replay's own spend (other runs on the shared ledger don't count against it).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from memobench.llm import EST_CHARS_PER_TOKEN, LLM, USAGE_FIELDS
from memsys import loop
from memsys.archive import when
from memsys.config import System, load
from memsys.dream import dream_due, dream_pass
from memsys.write import write_pass
from village.schema import Session, from_dict

SIDECARS = ("timeline.jsonl", "events.jsonl")
GIT_IDENTITY = ("-c", "user.name=memsys", "-c", "user.email=memsys@dream-village.invalid",
                "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null")  # fmt: skip


def load_sessions(path: Path | str) -> list[Session]:
    """Session dicts from a .json file (one dict or a list), a .jsonl file, or a directory of them. A directory is
    read one level deep only: generator subfolders such as attempt1/ and ab_haiku/ hold alternates, not sessions."""
    path = Path(path)
    files = sorted([*path.glob("*.json"), *path.glob("*.jsonl")]) if path.is_dir() else [path]
    sessions = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".jsonl":
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            items = data if isinstance(data, list) else [data]
        sessions += [from_dict(Session, d, f"{f.name}[{i}]") for i, d in enumerate(items)]
    return sessions


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", *GIT_IDENTITY, "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env=env).stdout  # fmt: skip


def _init_repo(repo: Path) -> None:
    if (repo / "timeline.jsonl").exists():
        raise FileExistsError(f"{repo} was already replayed; choose a new --out")
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    with open(repo / ".git" / "info" / "exclude", "a", encoding="utf-8") as f:
        f.write("\n".join(SIDECARS) + "\n")
    for name in SIDECARS:
        (repo / name).touch()


def _commit(repo: Path, message: str, sim_time: datetime) -> str | None:
    """Commit every change in the memory tree at the sim time; None when nothing changed."""
    if not _git(repo, "status", "--porcelain"):
        return None
    _git(repo, "add", "-A")
    stamp = sim_time.isoformat() + "+00:00"
    env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    _git(repo, "commit", "-q", "-m", message, env=env)
    return _git(repo, "rev-parse", "HEAD").strip()


def _append(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _until_end(until: str | None) -> datetime:
    """--until as an inclusive bound: a date means through the end of that day."""
    if until is None:
        return datetime.max
    return when(until) + timedelta(days=1) - timedelta(microseconds=1) if len(until) == 10 else when(until)


def replay_principal(
    llm: LLM, system: System, sessions: list[Session], repo: Path, *, until: str | None = None, salt: str | None = None
) -> dict:
    """One principal's chain, sequential. `sessions` are this principal's; returns a summary."""
    _init_repo(repo)
    store = system.store(repo)
    end = _until_end(until)
    mine = sorted((s for s in sessions if when(s.start) <= end), key=lambda s: when(s.start))
    summary = {"sessions": len(mine), "writes": 0, "dreams": 0, "commits": 0, "refusals": 0, "errors": 0,
               "cost_usd": 0.0, "tokens": dict.fromkeys(USAGE_FIELDS, 0)}  # fmt: skip

    def record(event: str, sim_time: datetime, session_id: str | None, chat: loop.Chat) -> None:
        message = f"{event} {session_id or sim_time.isoformat()}\n\n{chat.text or ('refused' if chat.refused else '')}"
        sha = _commit(repo, message, sim_time)
        summary[event + "s"] += 1
        summary["commits"] += sha is not None
        summary["refusals"] += chat.refused
        summary["errors"] += chat.error is not None
        summary["cost_usd"] += chat.cost_usd
        for k in USAGE_FIELDS:
            summary["tokens"][k] += chat.usage[k]
        common = {"sim_time": sim_time.isoformat(), "event": event, "session_id": session_id,
                  "cost_usd": chat.cost_usd, "tokens": dict(chat.usage)}  # fmt: skip
        _append(repo / "events.jsonl", {**common, "sha": sha, "hops": chat.hops, "edits": chat.edits,
                                        "tool_errors": sum(not c["ok"] for c in chat.calls), "refused": chat.refused,
                                        "stop_reason": chat.stop_reason, "error": chat.error, "text": chat.text})  # fmt: skip
        if sha is not None:
            _append(repo / "timeline.jsonl", {"sha": sha, **common, "config_sha256": system.sha256,
                                              "models": {event: system.setting("models", event)}})  # fmt: skip

    def maybe_dream(last: str | None, pending: int, now: datetime) -> int:
        at = dream_due(system, last, pending, now)
        if at is None:
            return pending
        record("dream", at, None, dream_pass(llm, system, store, at, archive=mine, salt=salt))
        return 0

    last, pending = None, 0
    for s in mine:
        pending = maybe_dream(last, pending, when(s.start))
        record("write", when(s.start), s.id, write_pass(llm, system, store, s, archive=mine, salt=salt))
        last, pending = s.start, pending + 1
    maybe_dream(last, pending, end)
    return summary


def replay(
    llm: LLM,
    system: System,
    sessions: list[Session],
    out: Path | str,
    *,
    principals: list[str] | None = None,
    until: str | None = None,
    salt: str | None = None,
) -> dict:
    """Replay every principal in parallel into <out>/ and write <out>/replay_manifest.json (returned)."""
    out = Path(out)
    t0, started = time.monotonic(), datetime.now().isoformat(timespec="seconds")
    by_principal: dict[str, list[Session]] = {}
    for s in sessions:
        if principals is None or s.principal in principals:
            by_principal.setdefault(s.principal, []).append(s)
    with ThreadPoolExecutor(max_workers=max(1, len(by_principal))) as pool:
        futures = {p: pool.submit(replay_principal, llm, system, ss, out / p, until=until, salt=salt)
                   for p, ss in sorted(by_principal.items())}  # fmt: skip
    summaries = {p: f.result() for p, f in futures.items()}  # re-raises a failed chain, after all chains ended

    end = _until_end(until)
    replayed = [s for ss in by_principal.values() for s in ss if when(s.start) <= end]
    chars = sum(len(turn.text) for s in replayed for turn in s.turns)
    cost = sum(v["cost_usd"] for v in summaries.values())
    tokens_est = chars / EST_CHARS_PER_TOKEN
    manifest = {
        "system": system.config["system"],
        "config_path": str(system.path),
        "config_sha256": system.sha256,
        "prompt_sha256": system.prompt_sha256,
        "models": system.models,
        "llm_run_id": llm.run_id,
        "salt": salt,
        "until": until,
        "principals": summaries,
        "sessions": sum(v["sessions"] for v in summaries.values()),
        "commits": sum(v["commits"] for v in summaries.values()),
        "refusals": sum(v["refusals"] for v in summaries.values()),
        "cost_usd": cost,
        "tokens": {k: sum(v["tokens"][k] for v in summaries.values()) for k in USAGE_FIELDS},
        "session_chars": chars,
        "session_tokens_est": round(tokens_est),  # chars / EST_CHARS_PER_TOKEN; an estimate, not a tokenizer count
        "cost_per_1m_session_tokens_est": cost / tokens_est * 1e6 if tokens_est else 0.0,
        "started_at": started,
        "wall_s": time.monotonic() - t0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def own_budget(llm: LLM, allowance: float) -> None:
    """Cap this replay at `allowance` USD of its OWN spend: before every call, re-base the LLM's ledger-wide budget so
    other runs appending to the shared ledger can't eat this replay's allowance."""
    start, call = llm.spent_usd, llm.call

    def guarded(req):
        llm.budget_usd = llm.total_usd + max(0.0, allowance - (llm.spent_usd - start))
        return call(req)

    llm.call = guarded


def checkout(out: Path | str, principal: str, t: str | datetime) -> Path:
    """A read-only fork: a fresh temp dir with the memory at the last commit with sim_time <= t."""
    repo = Path(out) / principal
    lines = (repo / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    eligible = [r for r in map(json.loads, filter(str.strip, lines)) if when(r["sim_time"]) <= when(t)]
    dest = Path(tempfile.mkdtemp(prefix=f"memsys-{principal}-"))
    if eligible:
        tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", eligible[-1]["sha"]],
                             capture_output=True, check=True).stdout  # fmt: skip
        with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
            archive.extractall(dest, filter="data")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memsys.replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", required=True, help="a sessions .json/.jsonl file or a directory of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--principals", help="comma-separated principal ids (default: all)")
    ap.add_argument("--until", help="last sim date (YYYY-MM-DD, inclusive) or datetime")
    ap.add_argument("--salt", help="sets metadata.user_id=replay-<salt>: a new cache key, the same prompt (A/A runs)")
    ap.add_argument("--budget", type=float, default=5.0, help="USD this replay may spend (default 5)")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    system = load(args.config)
    llm = LLM(run_id=f"memsys-replay-{Path(args.out).name}", allowed_models=tuple(sorted(set(system.models.values()))))
    own_budget(llm, args.budget)
    manifest = replay(llm, system, load_sessions(args.sessions), Path(args.out),
                      principals=args.principals.split(",") if args.principals else None,
                      until=args.until, salt=args.salt)  # fmt: skip
    print(json.dumps({k: manifest[k] for k in ("sessions", "commits", "refusals", "cost_usd", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
