"""village.schema: dataclass <-> JSON conversion and the structured-outputs schemas derived from the dataclasses."""

from __future__ import annotations

import jsonschema
import pytest

from village.schema import (
    Bible, DisclosureItem, Episode, Fact, Impression, Person, Reference, Usage, Voice, from_dict, json_schema, load,
    dump, to_dict,
)  # fmt: skip


def fact(**kw) -> Fact:
    base = dict(id="kai.f01", subject="kai", attribute="job", value="barista", valid_from=None, valid_to="2025-09-08",
                superseded_by="kai.f02", class_="useful")  # fmt: skip
    base.update(kw)
    return Fact(**base)


def bible() -> Bible:
    imp = Impression(
        id="kai.i01", statement="Downplays her own wins", about="kai",
        evidence=[Episode("kai.i01.e1", "2025-08-27", "cuts the award line from her essay")],
        formed=None, changed=None, changed_to="", decoy=False, sensitive=False,
    )  # fmt: skip
    return Bible(
        principal="kai", summary="CC student", facts=[fact()], impressions=[imp],
        references=[Reference("kai.r01", "the usual", "oat latte, extra shot", "2025-08-26")], arcs=[],
        voice=Voice("lowercase, no periods", "blunt", ["says 'lowkey'"], ["ok so"]),
        usage=Usage("daily", [7, 22]),
    )  # fmt: skip


def test_fact_class_field_is_written_as_class():
    d = to_dict(fact())
    assert d["class"] == "useful" and "class_" not in d


def test_nested_bible_round_trips_through_json():
    b = bible()
    assert from_dict(Bible, to_dict(b)) == b


def test_dump_then_load_round_trips(tmp_path):
    b = bible()
    dump(tmp_path / "x" / "bible.json", b)
    assert load(tmp_path / "x" / "bible.json", Bible) == b


def test_from_dict_rejects_unknown_field():
    d = to_dict(fact()) | {"confidence": 0.9}
    with pytest.raises(ValueError, match="confidence"):
        from_dict(Fact, d)


def test_from_dict_rejects_missing_required_field():
    d = to_dict(fact())
    del d["value"]
    with pytest.raises(ValueError, match="value"):
        from_dict(Fact, d)


def test_from_dict_allows_missing_field_with_default():
    d = {"session_id": "kai.s01", "item_id": "kai.f01", "mode": "stated", "task_type": "tutoring", "note": ""}
    assert from_dict(DisclosureItem, d).forget_request is False


def test_from_dict_rejects_value_outside_literal():
    d = to_dict(fact()) | {"class": "maybe"}
    with pytest.raises(ValueError, match="maybe"):
        from_dict(Fact, d)


def test_from_dict_rejects_wrong_scalar_type():
    d = {"id": "kai", "name": "Kai", "tier": "principal", "age": "nineteen", "role": "", "summary": "", "traits": []}
    with pytest.raises(ValueError, match="age"):
        from_dict(Person, d)


def test_from_dict_rejects_null_for_non_optional_field():
    d = to_dict(fact()) | {"value": None}
    with pytest.raises(ValueError, match="value"):
        from_dict(Fact, d)


def _objects(schema: dict):
    """Every object node in a schema."""
    if schema.get("type") == "object":
        yield schema
    for v in schema.values():
        if isinstance(v, dict):
            yield from _objects(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    yield from _objects(x)


def test_schema_closes_every_object_and_requires_every_field():
    """Structured outputs reject an object without additionalProperties: false."""
    objs = list(_objects(json_schema(Bible)))
    assert len(objs) >= 7  # Bible, Fact, Impression, Episode, Reference, Arc, Beat, Voice, Usage
    for o in objs:
        assert o["additionalProperties"] is False
        assert sorted(o["required"]) == sorted(o["properties"])


def test_schema_accepts_a_real_record_and_rejects_a_bad_class():
    schema = json_schema(Bible)
    good = to_dict(bible())
    jsonschema.validate(good, schema)
    bad = to_dict(bible())
    bad["facts"][0]["class"] = "maybe"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_schema_allows_null_only_for_optional_fields():
    props = json_schema(Fact)["properties"]
    jsonschema.validate(None, props["valid_to"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(None, props["value"])


def test_schema_omit_drops_code_filled_fields():
    schema = json_schema(Fact, omit={Fact: {"id"}})
    assert "id" not in schema["properties"] and "id" not in schema["required"]
