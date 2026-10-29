"""The always-loaded context blocks: <profile>, <preferences>, <memory_listing>, and (for the dream) every body."""

from __future__ import annotations

from memsys.store import INJECTED, Store, version_of


def _tag(store: Store, rel: str, tag: str, versions: bool) -> str:
    text = store.text("/" + rel)
    attr = f' version="{version_of(text)}"' if versions and text else ""
    body = text.rstrip("\n") if text else "(empty)"
    return f"<{tag}{attr}>\n{body}\n</{tag}>"


def injected(store: Store, *, versions: bool) -> str:
    """Profile, preferences (with version tokens when the step may edit them) and the listing."""
    return "\n".join(
        [_tag(store, "profile.md", "profile", versions), _tag(store, "preferences.md", "preferences", versions),
         store.listing()]
    )  # fmt: skip


def memory_files(store: Store) -> str:
    """<memory_files>: the full text and version of every file except profile and preferences."""
    files = [p for p in store.files() if p.lstrip("/") not in INJECTED]
    blocks = []
    for r in store.read(files):
        blocks.append(f'<file path="{r["path"]}" version="{r["version"]}">\n{r["content"].rstrip(chr(10))}\n</file>')
    return "<memory_files>\n" + ("\n".join(blocks) if blocks else "(none)") + "\n</memory_files>"
