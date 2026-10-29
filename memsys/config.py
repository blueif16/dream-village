"""System definition = one config JSON + the prompt files it names (system-v0.md §2, D19).

Every setting in the §2 table is a key in `config/v0.json`, so a candidate change is one diff to the config or to one
prompt file. A setting whose alternative values need new code is listed in IMPLEMENTED; any other value raises
NotImplementedError at load time instead of being silently ignored.

VARIANTS are the settings with more than one implemented value (round-1 candidates, T5). A key absent from the
config means its v0 value, so v0.json stays byte-identical. A non-v0 value is carried by a prompt variant that opens
a marker section (e.g. `<first_mention>`); the config loads only if each marker is present exactly when its value is
set, so a setting can never silently disagree with its prompt. Some values also change code: `forget: redact_raw`
adds the forget_span tool to the write pass, `dream_inference: patterns` adds archive_search to the dream.

`System.sha256` hashes the canonical config JSON plus every prompt's text: it is the `config_sha256` the replay
timeline records, so a prompt edit changes it and a reformatted config does not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from memsys.store import Store

V0 = Path(__file__).resolve().parent / "config" / "v0.json"
STEPS = ("write", "dream", "answer")

PROMPT_SETTINGS = "its behaviour lives in the prompt files; point `prompts` at a new prompt file instead"
IMPLEMENTED: dict[str, tuple[tuple, str]] = {
    "write_after": (
        ("session",),
        "sessions are fixed scripts (D27), so same-session recall is served by the live context, and a per-turn "
        "write pass would multiply write cost by the turn count; only 'session' is implemented",
    ),
    "consolidation_cadence": (("nightly",), "only the nightly cadence is implemented"),
    "consolidation_style": (("small_edits",), "only small edits (str_replace/append/delete) are implemented"),
    "file_picker": (("answer_model",), "only the answering model picks files, by memory_read"),
    "file_organization": (("claude_apps",), "only the Claude apps layout is implemented (D23)"),
    "stale_values": (("inline_previously",), PROMPT_SETTINGS),
    "inference_policy": (("inferred_non_sensitive",), PROMPT_SETTINGS),
}

# setting -> (v0 value, {value: (step, marker section) or None for the v0 value})
VARIANTS: dict[str, tuple[str, dict[str, tuple[str, str] | None]]] = {
    "answer_read_policy": ("as_needed", {"as_needed": None, "read_first": ("answer", "<read_memory_first>")}),
    "write_threshold": ("recurs_or_remember", {"recurs_or_remember": None, "first_mention": ("write", "<first_mention>")}),
    "profile_refresh": ("as_needed", {"as_needed": None, "every_session": ("write", "<profile_every_session>")}),
    "dream_inference": ("none", {"none": None, "patterns": ("dream", "<inferred_patterns>")}),
    "forget": ("memory_only", {"memory_only": None, "redact_raw": ("write", "<forget_in_history>")}),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class System:
    path: Path
    config: dict
    prompts: dict[str, str]

    @property
    def prompt_sha256(self) -> dict[str, str]:
        return {step: sha256_text(text) for step, text in sorted(self.prompts.items())}

    @property
    def sha256(self) -> str:
        canonical = json.dumps({"config": self.config, "prompts": self.prompt_sha256}, sort_keys=True, separators=(",", ":"))
        return sha256_text(canonical)

    @property
    def models(self) -> dict[str, str]:
        return dict(self.config["models"])

    def get(self, key: str) -> str:
        """A VARIANTS setting's value (its v0 value when the config omits the key)."""
        return self.config.get(key, VARIANTS[key][0])

    def setting(self, key: str, step: str):
        """A per-step setting such as models / effort / max_tokens / max_tool_hops."""
        return self.config[key][step]

    def store(self, root: Path | str) -> Store:
        """A principal's memory directory with this system's code-enforced limits."""
        return Store(
            root,
            profile_cap_words=self.config["profile_cap_words"],
            description_limit_chars=self.config["description_limit_chars"],
        )


def load(path: Path | str = V0) -> System:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, (allowed, why) in IMPLEMENTED.items():
        if config.get(key) not in allowed:
            raise NotImplementedError(f"{key}={config.get(key)!r} is not implemented: {why}")
    prompts = {step: (path.parent / config["prompts"][step]).read_text(encoding="utf-8") for step in STEPS}
    for key, (default, values) in VARIANTS.items():
        value = config.get(key, default)
        if value not in values:
            raise NotImplementedError(f"{key}={value!r} is not implemented: use one of {sorted(values)}")
        for other, marker in values.items():
            if marker is not None and (marker[1] in prompts[marker[0]]) != (value == other):
                raise ValueError(
                    f"{key}={value!r} disagrees with the {marker[0]} prompt: {marker[1]} must appear in it exactly "
                    f"when {key}={other!r}"
                )
    return System(path, config, prompts)
