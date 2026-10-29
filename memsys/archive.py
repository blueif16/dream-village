"""The raw archive: every past session, searched by keyword as a last resort (system-v0.md §1, D20: no embeddings).

Sim times are naive local ISO strings ("2025-08-25T07:40:00"); "before t" means start < t, so the session that is
happening at t is never in its own archive.

Forgotten spans (candidate C4, `forget: redact_raw`): the write pass records each span the user asked to forget in
<memory root>/.system/forgotten.jsonl, a committed sidecar that is not a memory file. `redact()` masks those spans in
the raw sessions the recent window and archive_search show; with no sidecar (v0) it changes nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from village.schema import Session

MASK = "[removed at the user's request]"
FORGOTTEN = Path(".system") / "forgotten.jsonl"  # under a principal's memory root; committed, never a memory file


def when(t: str | datetime) -> datetime:
    return t if isinstance(t, datetime) else datetime.fromisoformat(t)


def label(t: str | datetime) -> str:
    """'2025-09-10 19:30, Wednesday'."""
    return when(t).strftime("%Y-%m-%d %H:%M, %A")


def before(sessions: list[Session], t: str | datetime) -> list[Session]:
    """Sessions that started before t, oldest first."""
    cut = when(t)
    return sorted((s for s in sessions if when(s.start) < cut), key=lambda s: when(s.start))


def transcript(s: Session) -> str:
    turns = "\n\n".join(f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}" for turn in s.turns)
    return f'<session id="{s.id}" date="{label(s.start)}">\n{turns}\n</session>'


def _snippet(text: str, at: int, chars: int) -> str:
    start = max(0, at - chars // 3)
    end = min(len(text), start + chars)
    return ("…" if start else "") + text[start:end].replace("\n", " ") + ("…" if end < len(text) else "")


def archive_search(
    sessions: list[Session],
    t: str | datetime,
    query: str,
    *,
    max_sessions: int = 5,
    snippets_per_session: int = 3,
    snippet_chars: int = 300,
) -> str:
    """Case-insensitive keyword search over the sessions that started before t. A session scores one point per
    distinct query word found at the start of a word; the best (then newest) come first, each with its id, date and
    up to `snippets_per_session` matching snippets."""
    words = sorted({w for w in re.findall(r"\w+", query.lower()) if len(w) > 1})
    if not words:
        return "archive_search needs at least one keyword."
    patterns = [re.compile(r"\b" + re.escape(w), re.IGNORECASE) for w in words]
    hits = []
    for s in before(sessions, t):
        found: set[int] = set()
        snippets = []
        for turn in s.turns:
            matches = [(i, m) for i, p in enumerate(patterns) if (m := p.search(turn.text))]
            if not matches:
                continue
            found |= {i for i, _ in matches}
            if len(snippets) < snippets_per_session:
                who = "User" if turn.role == "user" else "Assistant"
                snippets.append(f"{who}: {_snippet(turn.text, min(m.start() for _, m in matches), snippet_chars)}")
        if found:
            hits.append((len(found), when(s.start), s, snippets))
    if not hits:
        return f"No past conversations match {query!r}."
    hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
    blocks = [
        f'<session id="{s.id}" date="{label(s.start)}">\n' + "\n".join(snips) + "\n</session>"
        for _, _, s, snips in hits[:max_sessions]
    ]
    return f"<archive_results query={query!r}>\n" + "\n".join(blocks) + "\n</archive_results>"


# ----------------------------------------------------------------------------- forgotten spans (candidate C4)


def _span(quote: str) -> re.Pattern:
    """The quote, case-insensitive, with any run of whitespace matching any other."""
    return re.compile(r"\s+".join(re.escape(w) for w in quote.split()), re.IGNORECASE)


def quote_in(session: Session, quote: str) -> bool:
    return bool(quote.split()) and any(_span(quote).search(turn.text) for turn in session.turns)


def load_forgotten(root: Path | str) -> list[dict]:
    path = Path(root) / FORGOTTEN
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_forgotten(root: Path | str, session_id: str, quote: str, at: str) -> None:
    path = Path(root) / FORGOTTEN
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"session_id": session_id, "quote": quote, "recorded_at": at}) + "\n")


def redact(sessions: list[Session], forgotten: list[dict]) -> list[Session]:
    """The sessions with every forgotten span replaced by MASK (sessions without one are returned as they are)."""
    spans: dict[str, list[re.Pattern]] = {}
    for f in forgotten:
        spans.setdefault(f["session_id"], []).append(_span(f["quote"]))
    out = []
    for s in sessions:
        if s.id not in spans:
            out.append(s)
            continue
        turns = []
        for turn in s.turns:
            text = turn.text
            for pattern in spans[s.id]:
                text = pattern.sub(MASK, text)
            turns.append(replace(turn, text=text))
        out.append(replace(s, turns=turns))
    return out
