"""A replay directory in memsys.replay's layout (<out>/<principal>/ repo + timeline.jsonl), built by hand."""

from __future__ import annotations

import json
import os
import subprocess

from tests import scorer_village as V

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", "-c", "commit.gpgsign=false"]


def make_replay(root, commits, principal: str = V.P):
    """commits: [(sim_time, event, session_id, {path: text, or None to delete})] -> root."""
    repo = root / principal
    repo.mkdir(parents=True)
    subprocess.run([*GIT, "-C", str(repo), "init", "-q"], check=True)
    (repo / ".git" / "info" / "exclude").write_text("timeline.jsonl\n")
    for sim_time, event, sid, files in commits:
        for path, text in files.items():
            target = repo / path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            if text is None:
                target.unlink()
            else:
                target.write_text(text)
        stamp = sim_time + "+00:00"
        env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        subprocess.run([*GIT, "-C", str(repo), "add", "-A"], check=True)
        subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", event], check=True, env=env)
        sha = subprocess.run([*GIT, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        with open(repo / "timeline.jsonl", "a") as f:
            f.write(json.dumps({"sha": sha, "sim_time": sim_time, "event": event, "session_id": sid, "cost_usd": 0.01,
                                "tokens": {}}) + "\n")  # fmt: skip
    return root
