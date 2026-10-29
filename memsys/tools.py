"""The memory tools the model calls, and the code that runs them against a Store (system-v0.md §3 Edits).

Each step offers a fixed tool list (so the cached prefix never changes): the write pass gets every memory tool plus
archive_search; the dream gets the edit tools, with memory_write limited to restructuring an existing file; the
answer gets memory_read and archive_search only, so answering can never change memory. Candidate settings add
one tool each: `forget: redact_raw` gives the write pass forget_span; `dream_inference: patterns` gives the dream
archive_search. archive_search always masks the spans recorded by forget_span (none in v0).
"""

from __future__ import annotations

from typing import Any

from memsys.archive import archive_search, load_forgotten, quote_in, record_forgotten, redact, when
from memsys.config import System
from memsys.store import Store, StoreError
from village.schema import Session

_PATH = {"type": "string", "description": "Memory path, e.g. /topics/food.md"}
_VERSION = {"type": "string", "description": "The version token from your latest memory_read of this file"}

SPECS: dict[str, dict] = {
    "memory_read": {
        "description": "Read one or more memory files. Returns each file's full text and its version token. "
        "Put every path you need in one call.",
        "input_schema": {
            "type": "object",
            "properties": {"paths": {"type": "array", "items": {"type": "string"}, "description": "Memory paths"}},
            "required": ["paths"],
        },
    },
    "memory_write": {
        "description": "Create a new memory file, or replace a whole existing file to restructure it. Replacing an "
        "existing file needs its version token. Use memory_str_replace or memory_append for small changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "content": {"type": "string", "description": "The complete file text, frontmatter included"},
                "version": {**_VERSION, "description": "Required when the file exists: its current version token"},
            },
            "required": ["path", "content"],
        },
    },
    "memory_str_replace": {
        "description": "Replace one exact piece of text in a memory file. old_str must appear exactly once; "
        "new_str may be empty to remove it. Needs the file's current version token.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "old_str": {"type": "string", "description": "Exact text to replace (unique in the file)"},
                "new_str": {"type": "string", "description": "Replacement text (may be empty)"},
                "version": _VERSION,
            },
            "required": ["path", "old_str", "new_str", "version"],
        },
    },
    "memory_append": {
        "description": "Add one line at the end of a memory file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": _PATH, "line": {"type": "string", "description": "One line, no newlines"}},
            "required": ["path", "line"],
        },
    },
    "memory_delete": {
        "description": "Delete a memory file. Only when the user asked to forget its whole subject, or after its "
        "lines were moved into another file.",
        "input_schema": {"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
    },
    "archive_search": {
        "description": "Case-insensitive keyword search over the user's earlier conversations. Returns matching "
        "session ids, dates and snippets. Use one or two distinctive words.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords"}},
            "required": ["query"],
        },
    },
    "forget_span": {
        "description": "Hide words the user asked you to forget from the raw conversation history. Give the id of "
        "the session that contains them and the exact words, copied from it. Call once per passage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "The session that contains the words, e.g. kai.s03"},
                "quote": {"type": "string", "description": "The exact words to hide, copied from that session"},
            },
            "required": ["session_id", "quote"],
        },
    },
}

WRITE_TOOLS = ("memory_read", "memory_write", "memory_str_replace", "memory_append", "memory_delete", "archive_search")
DREAM_TOOLS = ("memory_read", "memory_write", "memory_str_replace", "memory_append", "memory_delete")
ANSWER_TOOLS = ("memory_read", "archive_search")
EDITS = {"memory_write", "memory_str_replace", "memory_append", "memory_delete", "forget_span"}


def write_tools(system: System) -> tuple[str, ...]:
    return WRITE_TOOLS + (("forget_span",) if system.get("forget") == "redact_raw" else ())


def dream_tools(system: System) -> tuple[str, ...]:
    return DREAM_TOOLS + (("archive_search",) if system.get("dream_inference") == "patterns" else ())


def specs(names: tuple[str, ...]) -> list[dict]:
    return [{"name": n, **SPECS[n]} for n in names]


def render_reads(results: list[dict]) -> str:
    out = []
    for r in results:
        if "error" in r:
            out.append(f'<error path="{r["path"]}">{r["error"]}</error>')
        else:
            out.append(f'<file path="{r["path"]}" version="{r["version"]}">\n{r["content"]}</file>')
    return "\n".join(out)


class Toolbox:
    """Runs one tool call: (name, input) -> (result text, is_error, edited)."""

    def __init__(
        self,
        store: Store,
        names: tuple[str, ...],
        *,
        archive: list[Session] = (),
        now: str | None = None,
        search: dict | None = None,
        allow_create: bool | tuple[str, ...] = True,  # True, False, or the directories new files may go in
    ) -> None:
        self.store, self.names, self.archive, self.now = store, names, list(archive), now
        self.search = search or {}
        self.allow_create = allow_create

    def __call__(self, name: str, args: dict[str, Any]) -> tuple[str, bool, bool]:
        if name not in self.names:
            return f"unknown tool {name!r}; available: {', '.join(self.names)}", True, False
        try:
            return self._run(name, args), False, name in EDITS
        except StoreError as e:
            return str(e), True, False
        except (KeyError, TypeError, AttributeError) as e:
            return f"bad input for {name}: {e!r}; see the tool's input schema", True, False

    def _run(self, name: str, a: dict[str, Any]) -> str:
        s = self.store
        if name == "memory_read":
            paths = a["paths"]
            return render_reads(s.read([paths] if isinstance(paths, str) else list(paths)))
        if name == "archive_search":
            shown = redact(self.archive, load_forgotten(s.root))
            return archive_search(shown, self.now, a["query"], **self.search)
        if name == "forget_span":
            sid, quote = a["session_id"], a["quote"]
            match = [x for x in self.archive if x.id == sid and when(x.start) <= when(self.now)]
            if not match:
                raise StoreError(f"no session {sid!r} up to now; use an id from archive_search or this session")
            if not quote_in(match[0], quote):
                raise StoreError(f"those words are not in {sid}; copy the exact words from that session")
            record_forgotten(s.root, sid, quote, self.now)
            return f"hidden: the words will show as [removed at the user's request] in {sid}"
        if name == "memory_write":
            if self.allow_create is not True and not s.exists(a["path"]):
                if not self.allow_create:
                    raise StoreError("the dream only restructures existing files; it never creates one")
                if a["path"].strip().lstrip("/").split("/")[0] not in self.allow_create:
                    raise StoreError(f"the dream creates new files only under {', '.join(self.allow_create)}/")
            return f"wrote {a['path']}; version {s.write(a['path'], a['content'], a.get('version'))}"
        if name == "memory_str_replace":
            return f"edited {a['path']}; version {s.str_replace(a['path'], a['old_str'], a['new_str'], a['version'])}"
        if name == "memory_append":
            return f"appended to {a['path']}; version {s.append(a['path'], a['line'])}"
        if name == "memory_delete":
            s.delete(a["path"])
            return f"deleted {a['path']}"
        raise AssertionError(name)
