"""Dream Village records as plain dataclasses <-> JSON (design §3).

Dates are ISO strings ("2025-08-25"); session starts are naive local ISO timestamps ("2025-08-25T07:40:00").
A fact holds on [valid_from, valid_to): valid_to is the first day it no longer holds, None means open-ended.
Ids are stable and namespaced by who minted them: "cast.f03", "<principal>.f07", "<principal>.w2.f01",
"<principal>.i02", "<principal>.r01", "<principal>.s05".

A field whose name ends in "_" is written without it ("class_" -> "class", "with_" -> "with").
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

FactClass = Literal["useful", "trivia", "never_store", "forgotten"]
Tier = Literal["principal", "inner", "periphery"]
Mode = Literal["stated", "hinted"]
Role = Literal["user", "assistant"]
TaskType = Literal[
    "practical_guidance", "seeking_information", "writing_edit", "writing_new", "tutoring",
    "technical_help", "multimedia", "personal_reflection",
]  # fmt: skip
Frequency = Literal["daily", "several_per_week", "weekly"]


# --------------------------------------------------------------------------- world


@dataclass
class Place:
    id: str
    name: str
    kind: str
    description: str


@dataclass
class TownEvent:
    id: str
    date: str
    end_date: str | None
    title: str
    description: str
    places: list[str]


@dataclass
class World:
    town: str
    region: str
    summary: str
    texture: list[str]
    places: list[Place]
    events: list[TownEvent]


# --------------------------------------------------------------------------- facts (cast sheet, bibles, life logs)


@dataclass
class Fact:
    id: str
    subject: str
    attribute: str
    value: str
    valid_from: str | None
    valid_to: str | None
    superseded_by: str | None
    class_: FactClass


# --------------------------------------------------------------------------- cast


@dataclass
class Person:
    id: str
    name: str
    tier: Tier
    age: int
    role: str
    summary: str
    traits: list[str]


@dataclass
class Link:
    a: str
    b: str
    relation: str


@dataclass
class Cast:
    people: list[Person]
    links: list[Link]
    facts: list[Fact]  # the shared sheet: facts about non-principals, the same for every principal who knows them


# --------------------------------------------------------------------------- principal bible


@dataclass
class Episode:
    id: str
    date: str
    description: str


@dataclass
class Impression:
    id: str
    statement: str
    about: str
    evidence: list[Episode]
    formed: str | None
    changed: str | None
    changed_to: str
    decoy: bool
    sensitive: bool


@dataclass
class Reference:
    id: str
    term: str
    meaning: str
    first_use: str


@dataclass
class Beat:
    date: str
    what: str
    facts: list[str]


@dataclass
class Arc:
    id: str
    title: str
    summary: str
    beats: list[Beat]


@dataclass
class Voice:
    texting_style: str
    with_assistant: str
    quirks: list[str]
    sample_lines: list[str]


@dataclass
class Usage:
    frequency: Frequency
    active_hours: list[int]


@dataclass
class Bible:
    principal: str
    summary: str
    facts: list[Fact]
    impressions: list[Impression]
    references: list[Reference]
    arcs: list[Arc]
    voice: Voice
    usage: Usage


# --------------------------------------------------------------------------- life log


@dataclass
class LifeLogEntry:
    id: str
    date: str
    what: str
    with_: list[str]
    outcome: str
    feeling: str
    facts: list[str]
    impressions: list[str]


@dataclass
class LifeLogWeek:
    principal: str
    week: int
    entries: list[LifeLogEntry]
    new_facts: list[Fact]


# --------------------------------------------------------------------------- schedule, disclosure plan, sessions


@dataclass
class ScheduledSession:
    id: str
    principal: str
    start: str
    week: int
    task_type: TaskType


@dataclass
class DisclosureItem:
    session_id: str
    item_id: str
    mode: Mode
    task_type: TaskType
    note: str
    forget_request: bool = False  # this session is where the user asks the assistant to forget item_id
    never_store: bool = False  # D32(c): a never-store item the user states here; the memory must not keep it


@dataclass
class SessionPlan:
    session_id: str
    task_type: TaskType
    task: str
    items: list[DisclosureItem]


@dataclass
class Turn:
    role: Role
    text: str


@dataclass
class Session:
    id: str
    principal: str
    start: str
    turns: list[Turn]


@dataclass
class SessionFact:
    """D32(e): an unplanned detail a session's user turns state. It enters the fact log as `trivia`, so the integrity
    audit doesn't score a memory of it as fabricated."""

    source: str  # the session id
    quote: str  # the user's words, verbatim
    fact: Fact


# --------------------------------------------------------------------------- verification + recoverability


@dataclass
class Claim:
    text: str
    about: str
    kind: Literal["fact", "trait_stated", "trait_shown", "reference_use", "forget_request", "other"]
    item_id: str | None
    relation: Literal["consistent", "contradicts", "new"]
    speaker: Literal["user", "assistant"] = "user"
    quote: str = ""  # verbatim words of the turn that makes the claim (code confirms them in the transcript)
    support_quote: str = ""  # for an assistant claim: verbatim words of an earlier user turn that tell it this


@dataclass
class PlannedCheck:
    """The verifier's answer for one planned item: does it surface in a user turn, and where (verbatim)."""

    item_id: str
    present: bool
    quote: str


@dataclass
class SensitiveCheck:
    """Where a never-store item is stated: by the user, and (a leak) by the assistant, verbatim."""

    item_id: str
    user_quote: str
    assistant_quote: str


@dataclass
class Flag:
    kind: Literal["missing", "conflict", "trait_stated", "ungrounded", "leak", "refusal", "unparseable"]
    item_id: str | None
    detail: str


@dataclass
class Verification:
    session_id: str
    model: str
    attempt: int
    claims: list[Claim]
    flags: list[Flag]
    passed: bool
    checks: list[PlannedCheck] = field(default_factory=list)
    sensitive: list[SensitiveCheck] = field(default_factory=list)


@dataclass
class Inferred:
    statement: str
    strength: Literal["strong", "moderate", "weak"]
    sessions: list[str]


@dataclass
class Recoverability:
    impression_id: str
    evidence_sessions: list[str]
    inferred: list[Inferred]
    recovered: bool
    matched: str
    rationale: str


# --------------------------------------------------------------------------- JSON conversion


def _json_name(name: str) -> str:
    return name[:-1] if name.endswith("_") else name


def to_dict(obj: Any) -> Any:
    """A dataclass (or a list/dict of them) as plain JSON-ready data."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {_json_name(f.name): to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _convert(tp: Any, data: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(tp)
        if data is None:
            if type(None) not in options:
                raise ValueError(f"{where}: null is not allowed")
            return None
        return _convert(next(o for o in options if o is not type(None)), data, where)
    if origin is list:
        if not isinstance(data, list):
            raise ValueError(f"{where}: expected a list, got {type(data).__name__}")
        (item,) = typing.get_args(tp)
        return [_convert(item, x, f"{where}[{i}]") for i, x in enumerate(data)]
    if origin is Literal:
        if data not in typing.get_args(tp):
            raise ValueError(f"{where}: {data!r} not in {typing.get_args(tp)}")
        return data
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, data, where)
    if tp in (bool, int, str):
        if type(data) is not tp:  # bool is an int subclass; neither may stand in for the other
            raise ValueError(f"{where}: expected {tp.__name__}, got {data!r}")
        return data
    raise TypeError(f"{where}: unsupported type {tp!r}")


def from_dict(cls: type, data: Any, where: str | None = None) -> Any:
    """Build `cls` from JSON data; raises ValueError on a missing field, an unknown field or a wrong type."""
    where = where or cls.__name__
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected an object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {_json_name(f.name): f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown fields {unknown}")
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[f.name] = _convert(hints[f.name], data[name], f"{where}.{name}")
        elif f.default is dataclasses.MISSING:
            raise ValueError(f"{where}: missing field {name!r}")
    return cls(**kwargs)


def json_schema(tp: Any, *, omit: dict[type, set[str]] | None = None) -> dict:
    """The structured-outputs JSON schema of `tp`: every object closed, every field required, nullable only where the
    type is Optional. `omit` drops fields per dataclass (ones code fills in, such as the ids it assigns)."""
    omit = omit or {}
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        return {"anyOf": [json_schema(o, omit=omit) for o in typing.get_args(tp)]}
    if origin is list:
        return {"type": "array", "items": json_schema(typing.get_args(tp)[0], omit=omit)}
    if origin is Literal:
        return {"type": "string", "enum": list(typing.get_args(tp))}
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        names = [f.name for f in dataclasses.fields(tp) if f.name not in omit.get(tp, set())]
        return {
            "type": "object",
            "properties": {_json_name(n): json_schema(hints[n], omit=omit) for n in names},
            "required": [_json_name(n) for n in names],
            "additionalProperties": False,
        }
    simple = {str: "string", int: "integer", bool: "boolean", type(None): "null"}
    if tp in simple:
        return {"type": simple[tp]}
    raise TypeError(f"unsupported type {tp!r}")


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path: Path, cls: type) -> Any:
    return from_dict(cls, json.loads(path.read_text(encoding="utf-8")), path.name)


def load_list(path: Path, cls: type) -> list:
    return [from_dict(cls, x, f"{path.name}[{i}]") for i, x in enumerate(json.loads(path.read_text(encoding="utf-8")))]
