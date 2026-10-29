"""One principal's memory directory, in Claude apps' layout (system-v0.md §3), with the rules code enforces.

Layout: `profile.md`, `preferences.md`, `topics/<slug>.md`, `areas/<slug>.md`, `people/<slug>.md`. Paths are shown to
the model as "/topics/food.md"; a leading "/" is optional. Every read returns a version token (a content hash);
`write` over an existing file and `str_replace` need the current token ("read before write"). Every failed edit
raises StoreError with a message the model sees as its tool result, and leaves the file untouched.

Code-enforced rules (the rest are the prompts' job):
- profile.md is at most `profile_cap_words` words;
- topics/areas/people files carry frontmatter (name = path stem, description under `description_limit_chars`,
  optional sources and aliases lists); profile.md and preferences.md may omit it;
- one file per subject: no file's name or alias may equal another file's name or alias (case, dashes, underscores
  and spaces ignored).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

INJECTED = ("profile.md", "preferences.md")  # loaded into every context; not in the listing
DIRS = ("topics", "areas", "people")
PATH_RE = re.compile(r"^(?:profile\.md|preferences\.md|(?:topics|areas|people)/[a-z0-9][a-z0-9_-]*\.md)$")
KEYS = ("name", "description", "sources", "aliases")
LIST_KEYS = ("sources", "aliases")
WORD_RE = re.compile(r"\S*[^\W_]\S*")  # a whitespace-separated token with at least one letter or digit


class StoreError(Exception):
    """An edit the store refuses; str(e) is what the model is told."""


def version_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _subject_key(name: str) -> str:
    return re.sub(r"[\s_-]+", " ", name.strip().lower())


def split_frontmatter(text: str) -> tuple[dict | None, str]:
    """(frontmatter dict or None, body). Lists are written `[a, b]`; raises StoreError on a malformed block."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 3)
    if end < 0 or text[end + 4 : end + 5] not in ("", "\n"):
        raise StoreError("frontmatter is not closed: end it with a line '---'")
    meta: dict = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or key not in KEYS:
            raise StoreError(f"frontmatter line {line!r}: allowed keys are {', '.join(KEYS)}")
        if key in LIST_KEYS:
            if not (value.startswith("[") and value.endswith("]")):
                raise StoreError(f"frontmatter {key} must be a list like [a, b], got {value!r}")
            meta[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value.strip("'\"")
    return meta, text[end + 5 :]


class Store:
    def __init__(self, root: Path | str, *, profile_cap_words: int = 300, description_limit_chars: int = 150) -> None:
        self.root = Path(root)
        self.profile_cap_words = profile_cap_words
        self.description_limit_chars = description_limit_chars

    # ------------------------------------------------------------------ paths

    def _rel(self, path: str) -> str:
        rel = path.strip().lstrip("/")
        if not PATH_RE.match(rel):
            raise StoreError(
                f"invalid path {path!r}: use /profile.md, /preferences.md, or /topics|areas|people/<slug>.md "
                "with a lowercase slug of letters, digits, '-' or '_'"
            )
        return rel

    def files(self) -> list[str]:
        """Every memory file as "/<rel path>", sorted."""
        found = [p for p in INJECTED if (self.root / p).is_file()]
        for d in DIRS:
            found += [f"{d}/{p.name}" for p in sorted((self.root / d).glob("*.md")) if PATH_RE.match(f"{d}/{p.name}")]
        return sorted("/" + f for f in found)

    def exists(self, path: str) -> bool:
        return (self.root / self._rel(path)).is_file()

    def text(self, path: str) -> str:
        """The file's content, or "" when it doesn't exist."""
        p = self.root / self._rel(path)
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def meta(self, path: str) -> dict:
        try:
            meta, _ = split_frontmatter(self.text(path))
        except StoreError:
            return {}
        return meta or {}

    # ------------------------------------------------------------------ reads

    def read(self, paths: list[str]) -> list[dict]:
        """One entry per path: {"path", "content", "version"} or {"path", "error"}."""
        out = []
        for path in paths:
            try:
                rel = self._rel(path)
            except StoreError as e:
                out.append({"path": path, "error": str(e)})
                continue
            p = self.root / rel
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                out.append({"path": "/" + rel, "content": text, "version": version_of(text)})
            else:
                out.append({"path": "/" + rel, "error": "not found"})
        return out

    # ------------------------------------------------------------------ edits

    def write(self, path: str, content: str, version: str | None = None) -> str:
        """Create a file, or replace an existing one (needs its current version). Returns the new version."""
        rel = self._rel(path)
        if (self.root / rel).is_file():
            self._check_version(rel, version, "write")
        return self._save(rel, content)

    def str_replace(self, path: str, old: str, new: str, version: str | None) -> str:
        rel = self._rel(path)
        current = self._existing(rel)
        self._check_version(rel, version, "str_replace")
        n = current.count(old) if old else 0
        if n == 0:
            raise StoreError(f"old_str not found in /{rel}; read the file and copy the exact text")
        if n > 1:
            raise StoreError(f"old_str matches {n} places in /{rel}; include more surrounding text so it is unique")
        return self._save(rel, current.replace(old, new, 1))

    def append(self, path: str, line: str) -> str:
        """Add one line at the end of the file (created for profile.md / preferences.md if missing)."""
        rel = self._rel(path)
        line = line.rstrip("\n")
        if "\n" in line or not line.strip():
            raise StoreError("append takes one line of text (no newlines); use str_replace or write for more")
        current = self.text("/" + rel)
        if current and not current.endswith("\n"):
            current += "\n"
        return self._save(rel, current + line + "\n")

    def delete(self, path: str) -> None:
        rel = self._rel(path)
        self._existing(rel)
        (self.root / rel).unlink()

    # ------------------------------------------------------------------ rules

    def _existing(self, rel: str) -> str:
        p = self.root / rel
        if not p.is_file():
            raise StoreError(f"/{rel} not found")
        return p.read_text(encoding="utf-8")

    def _check_version(self, rel: str, version: str | None, op: str) -> None:
        current = version_of(self._existing(rel))
        if not version:
            raise StoreError(f"{op} on existing /{rel} needs its version: read the file first (current version {current})")
        if version != current:
            raise StoreError(f"stale version {version!r} for /{rel}: it changed since you read it; read it again")

    def _save(self, rel: str, content: str) -> str:
        self._validate(rel, content)
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return version_of(content)

    def _validate(self, rel: str, content: str) -> None:
        meta, body = split_frontmatter(content)
        stem = rel.rsplit("/", 1)[-1][: -len(".md")]
        if meta is None and rel not in INJECTED:
            raise StoreError(
                f"/{rel} needs frontmatter: a first line '---', then name: {stem}, description: <one line>, "
                "sources: [chat], aliases: [..], then '---'"
            )
        if rel == "profile.md" and word_count(body) > self.profile_cap_words:
            raise StoreError(
                f"profile.md would be {word_count(body)} words; the cap is {self.profile_cap_words}. Keep only what "
                "stays true for three months and move dated or 'currently' lines to /areas/ or /topics/"
            )
        if meta is None:
            return
        if meta.get("name") != stem:
            raise StoreError(f"frontmatter name must be the path stem {stem!r}, got {meta.get('name')!r}")
        description = meta.get("description", "")
        if not description:
            raise StoreError(f"/{rel} frontmatter needs a one-line description")
        if len(description) >= self.description_limit_chars:
            raise StoreError(
                f"description is {len(description)} characters; it must be under {self.description_limit_chars}"
            )
        mine = {_subject_key(n): n for n in [stem, *meta.get("aliases", [])]}
        for other in self.files():
            if other == "/" + rel:
                continue
            theirs = {_subject_key(n) for n in [self.meta(other).get("name", ""), *self.meta(other).get("aliases", [])]}
            clash = sorted(n for k, n in mine.items() if k and k in theirs)
            if clash:
                raise StoreError(
                    f"{other} already covers {', '.join(clash)} (one file per subject): edit {other} instead, "
                    "and add any new name to its aliases"
                )

    # ------------------------------------------------------------------ rendering

    def listing(self) -> str:
        """<memory_listing>: one line per file except profile/preferences (path, description, aliases, sources)."""
        lines = []
        for path in self.files():
            if path.lstrip("/") in INJECTED:
                continue
            m = self.meta(path)
            lines.append(
                f"- {path} | {m.get('description', '')} | aliases: [{', '.join(m.get('aliases', []))}]"
                f" | sources: [{', '.join(m.get('sources', []))}]"
            )
        body = "\n".join(lines) if lines else "(no memory files yet)"
        return f"<memory_listing>\n{body}\n</memory_listing>"
