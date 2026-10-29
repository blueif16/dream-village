"""village.spec: the pilot spec reproduces pilot-v1's requests byte for byte (so its response cache still answers them),
and the full-village spec follows design §2 (12 principals by age and usage, every principal linked)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date

from village import generate as g
from village.spec import FULL, PILOT

# sha256 of the prompt files pilot-v1 ran with (main, merge of feat/dream-village-generator).
PILOT_V1_PROMPTS = {
    "world": "27080c68e4136d41fdcc2166ef0409025a85121dea623a986f2e4c62a06ebffd",
    "cast": "44a7d90f18478917821695f655114e409fce03de370efe3f3661a35df9f3856c",
    "bible": "0acfc5d1c5087da2c30eefcea1bbceef94618cde5a7ddd8c8971b4319189996f",
    "life_log": "7dbcc92d91fca855669b84db82259ea6816d4a225906b208aa6e9a0070986d0f",
}
PILOT_V1_R1_KEY = "f694cf2225934c6b62e953bd724216103210b2886770176ac93d18b98d6f0959"


def test_pilot_prompts_render_byte_identical_to_pilot_v1():
    for name, sha in PILOT_V1_PROMPTS.items():
        assert hashlib.sha256(g.prompt(name, PILOT).encode()).hexdigest() == sha, name


def test_pilot_world_request_is_the_one_pilot_v1_cached():
    assert g.r1_request(PILOT).key == PILOT_V1_R1_KEY


def test_full_prompts_fill_every_placeholder_and_differ_from_the_pilot():
    for name in ("world", "cast", "bible"):
        text = g.prompt(name, FULL)
        assert not re.search(r"\$[a-z_]+", text), name
        assert text != g.prompt(name, PILOT), name


def test_full_village_cast_follows_design_section_2():
    ages = Counter(next(band for band, (lo, hi) in {"15-17": (15, 17), "18-22": (18, 22), "23-30": (23, 30),
                                                    "30s": (30, 39), "40s": (40, 49)}.items()
                        if lo <= p.ages[0] and p.ages[1] <= hi) for p in FULL.principals)  # fmt: skip
    assert ages == Counter({"15-17": 2, "18-22": 4, "23-30": 3, "30s": 2, "40s": 1})
    assert Counter(p.frequency for p in FULL.principals) == Counter({"daily": 4, "several_per_week": 5, "weekly": 3})


def test_every_full_village_principal_is_linked_to_another():
    linked = {i for pair in FULL.required_links for i in pair}
    assert linked == set(range(1, len(FULL.principals) + 1))


def test_full_village_horizon_is_twelve_months_from_the_first_school_week():
    days = g.horizon_days(FULL.start, FULL.weeks)
    assert days[0] == date(2025, 8, 25) and len(days) == 364


def test_pilot_horizon_has_only_labor_day():
    assert g.holidays(g.horizon_days(PILOT.start, PILOT.weeks)) == {date(2025, 9, 1): "Labor Day (US federal holiday)"}


def test_full_year_has_every_us_federal_holiday():
    got = g.holidays(g.horizon_days(FULL.start, FULL.weeks))
    expected = {
        date(2025, 9, 1): "Labor Day", date(2025, 10, 13): "Columbus Day / Indigenous Peoples' Day",
        date(2025, 11, 11): "Veterans Day", date(2025, 11, 27): "Thanksgiving Day", date(2025, 12, 25): "Christmas Day",
        date(2026, 1, 1): "New Year's Day", date(2026, 1, 19): "Martin Luther King Jr. Day",
        date(2026, 2, 16): "Presidents' Day", date(2026, 5, 25): "Memorial Day", date(2026, 6, 19): "Juneteenth",
        date(2026, 7, 4): "Independence Day",
    }  # fmt: skip
    assert got == {d: f"{name} (US federal holiday)" for d, name in expected.items()}


# --------------------------------------------------------------------------- full13: 12 principals x 13 weeks (Shiran)


def test_full13_is_the_full_cast_over_the_first_13_weeks():
    from village.spec import FULL13, VILLAGES

    days = g.horizon_days(FULL13.start, FULL13.weeks)
    assert (days[0], days[-1], len(days)) == (date(2025, 8, 25), date(2025, 11, 23), 91)
    assert FULL13.principals == FULL.principals and FULL13.required_links == FULL.required_links
    assert VILLAGES["full13"] is FULL13


def test_full13_prompts_are_sized_for_a_quarter():
    from village.spec import FULL13

    for name in ("world", "cast", "bible"):
        text = g.prompt(name, FULL13)
        assert not re.search(r"\$[a-z_]+", text), name
        assert text != g.prompt(name, FULL), name
    assert FULL13.bible_bar.impressions[1] < FULL.bible_bar.impressions[1]
    assert FULL13.events_min < FULL.events_min


def test_full13_horizon_holidays():
    from village.spec import FULL13

    got = g.holidays(g.horizon_days(FULL13.start, FULL13.weeks))
    assert sorted(d.isoformat() for d in got) == ["2025-09-01", "2025-10-13", "2025-11-11"]  # Thanksgiving is 11-27


# --------------------------------------------------------------------------- heldout: the second, smaller town (D16)


def _heldout():
    from village.spec import VILLAGES

    spec = VILLAGES.get("heldout")
    assert spec is not None, "no heldout village spec"
    return spec


def test_heldout_is_a_smaller_town_over_the_dev_horizon():
    """D16: 4-6 principals; the same 13 weeks as full13, so lag strata compare across the two towns."""
    from village.spec import FULL13

    spec = _heldout()
    assert 4 <= len(spec.principals) <= 6
    assert g.horizon_days(spec.start, spec.weeks) == g.horizon_days(FULL13.start, FULL13.weeks)


def test_every_heldout_principal_is_linked_to_another():
    spec = _heldout()
    linked = {i for pair in spec.required_links for i in pair}
    assert linked == set(range(1, len(spec.principals) + 1))


def test_heldout_keeps_the_dev_usage_mix_and_a_minor():
    """Every usage tier appears (D15), and a minor, so the minors-without-age rule (C2) is tested on held-out too."""
    spec = _heldout()
    assert {p.frequency for p in spec.principals} == {"daily", "several_per_week", "weekly"}
    assert any(p.minor for p in spec.principals)
    assert all(p.minor == (p.ages[1] < 18) for p in spec.principals)


def test_heldout_principals_are_not_the_dev_cast():
    """A held-out resident who is a dev resident's description would let the loop tune to them."""
    spec = _heldout()
    assert not {p.life for p in spec.principals} & {p.life for p in FULL.principals}


def test_heldout_cast_prompt_is_sized_for_its_principals():
    from village.spec import FULL13

    spec = _heldout()
    for name in ("world", "cast", "bible"):
        assert not re.search(r"\$[a-z_]+", g.prompt(name, spec)), name
    tiers = lambda s: [tuple(map(int, m)) for m in re.findall(r"(?:bosses|manager's manager): (\d+)-(\d+)", g.prompt("cast", s))]  # noqa: E731
    (inner, periphery), (dev_inner, dev_periphery) = tiers(spec), tiers(FULL13)
    assert inner[1] < dev_inner[0] and periphery[1] < dev_periphery[0], "a 5-principal town asks for a 12-principal cast"


# The full13-v1 world request, as logged in data/villages/full13-v1/llm/logs/requests.jsonl (its cache holds it).
FULL13_V1_R1_KEY = "270c81d20e616bff19d72e582af5a68491160692c6f9e6376bf50a8d11ec2267"


def _dev_village(tmp_path):
    (tmp_path / "world.json").write_text(json.dumps({
        "town": "Alcaraz Flats", "region": "southern edge of invented Tamsen County",
        "places": [{"id": "pl-a", "name": "Calle Ortega"}, {"id": "pl-b", "name": "Taquería Colima"}]}))  # fmt: skip
    (tmp_path / "cast.json").write_text(json.dumps({"people": [{"id": "isabel", "name": "Isabel Tolentino"},
                                                               {"id": "grace", "name": "Grace Tolentino"}]}))  # fmt: skip
    return tmp_path


def test_full13_world_request_is_the_one_full13_v1_cached():
    from village.spec import FULL13

    assert g.r1_request(FULL13).key == FULL13_V1_R1_KEY


def test_heldout_is_generated_distinct_from_the_dev_village():
    assert _heldout().distinct_from == "data/villages/full13-v1"


def test_heldout_world_request_rules_out_the_dev_town_and_its_places(tmp_path):
    """heldout-v1's first world came back as the dev town (same name): the request must name what to avoid."""
    import dataclasses

    spec = dataclasses.replace(_heldout(), distinct_from=str(_dev_village(tmp_path)))
    user = g.r1_request(spec).params["messages"][0]["content"]
    for name in ("Alcaraz Flats", "Tamsen County", "Calle Ortega", "Taquería Colima"):
        assert name in user, name


def test_heldout_cast_request_rules_out_the_dev_cast_names(tmp_path):
    import dataclasses

    from village.schema import Place, World

    spec = dataclasses.replace(_heldout(), distinct_from=str(_dev_village(tmp_path)))
    world = World("Other Town", "", "", [], [Place("pl-x", "Marsh Light Coffee", "cafe", "")], [])
    user = g.r2_request(spec, world).params["messages"][0]["content"]
    assert "Isabel Tolentino" in user and "Grace Tolentino" in user
