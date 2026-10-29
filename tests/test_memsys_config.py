"""memsys.config: v0 settings and prompts as data (system-v0.md §2), so each candidate is one diff."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from memsys.config import V0, load

ROOT = Path(__file__).resolve().parent.parent / "memsys"


def copy_system(tmp_path: Path) -> Path:
    """A copy of config/ + prompts/ so a test can edit one file."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    return tmp_path / "config" / "v0.json"


def edit_config(path: Path, **changes) -> None:
    cfg = json.loads(path.read_text())
    cfg.update(changes)
    path.write_text(json.dumps(cfg))


def test_v0_holds_the_system_v0_settings_table():
    cfg = load(V0).config
    assert cfg["consolidation_cadence"] == "nightly" and cfg["consolidation_time"] == "03:00"
    assert cfg["consolidation_style"] == "small_edits"
    assert cfg["profile_cap_words"] == 300
    assert cfg["recent_window_sessions"] == 10
    assert cfg["write_after"] == "session"
    assert cfg["models"] == {"write": "claude-haiku-5-5", "dream": "claude-haiku-5-5", "answer": "claude-sonnet-5-5"}
    for setting in ("file_picker", "file_organization", "write_threshold", "stale_values", "inference_policy"):
        assert setting in cfg


def test_write_after_turn_is_not_implemented_and_says_why(tmp_path):
    path = copy_system(tmp_path)
    edit_config(path, write_after="turn")
    with pytest.raises(NotImplementedError, match="(?i)fixed script"):
        load(path)


def test_an_unimplemented_setting_value_is_refused_by_name(tmp_path):
    path = copy_system(tmp_path)
    edit_config(path, consolidation_cadence="idle_6h")
    with pytest.raises(NotImplementedError, match="consolidation_cadence"):
        load(path)


def test_system_hash_covers_prompt_text(tmp_path):
    path = copy_system(tmp_path)
    before = load(path).sha256
    prompt = tmp_path / "prompts" / "dream.md"
    prompt.write_text(prompt.read_text() + "\nOne more rule.\n")
    assert load(path).sha256 != before


def test_system_hash_ignores_config_formatting(tmp_path):
    path = copy_system(tmp_path)
    before = load(path).sha256
    path.write_text(json.dumps(json.loads(path.read_text()), indent=7, sort_keys=False))
    assert load(path).sha256 == before


def test_write_prompt_carries_the_full_never_store_list():
    text = load(V0).prompts["write"].lower()
    for rule in ("government-id", "financial-account", "immigration", "caste", "minor", "date of birth",
                 "sexual history", "abuse", "criminal", "suicide", "self-harm", "disordered eating"):  # fmt: skip
        assert rule in text, rule
