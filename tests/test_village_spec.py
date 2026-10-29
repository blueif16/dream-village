"""village.spec: the pilot spec reproduces pilot-v1's requests byte for byte (so its response cache still answers them),
and the full-village spec follows design §2 (12 principals by age and usage, every principal linked)."""

from __future__ import annotations

import hashlib
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
