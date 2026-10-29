"""Village specs: who the principals are, the horizon, and the counts the world/cast/bible prompts ask for.

PILOT reproduces pilot-v1's requests byte for byte (its prompt values are pilot-v1's text, frozen), so pilot-v1's
response cache still answers its world, cast, bible and life-log stages. FULL is the dev village of design §2:
12 principals over 12 months.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Principal:
    ages: tuple[int, int]
    frequency: str  # daily | several_per_week | weekly
    life: str
    minor: bool = False


@dataclass(frozen=True)
class BibleBar:
    """What parse_bible enforces (a little looser than the prompt asks, so near-misses aren't regenerated)."""

    impressions: tuple[int, int]
    decoys: int
    episodes_min: int
    references_min: int
    arcs: tuple[int, int]
    forgotten: tuple[int, int]
    changes_min: int


@dataclass(frozen=True)
class VillageSpec:
    name: str
    start: date
    weeks: int
    principals: tuple[Principal, ...]
    links: str  # the cast request's sentence on how the principals are linked
    required_links: tuple[tuple[int, int], ...]  # 1-based principal pairs the cast must link
    prompt_vars: dict[str, str]
    bible_bar: BibleBar
    events_min: int
    never_store_per_month: int = 2  # D32(c): 1-3 task-driven never-store disclosures per principal per month
    max_tokens: dict[str, int] = field(default_factory=dict)  # per-stage overrides of generate.STAGES
    distinct_from: str | None = None  # a generated village dir whose town and cast names this village must not reuse


# pilot-v1's prompt text, verbatim (do not edit: it keys pilot-v1's response cache)
PILOT_PROMPT_VARS = {
    'cast_changes': '2-4',
    'cast_never_store': '0-2',
    'changes': '  - Changes: at least 3 facts change inside the horizon because of the arcs (end the old fact on the change date,\n    set its `superseded_by` to the new fact, start the new fact that day). Put change dates between 2025-08-27 and\n    2025-09-14 so later conversations can mention them.',
    'count_bar': '1. Counts inside the ranges above: facts, at least 3 in-horizon changes, 1-2 never_store, exactly 1 forgotten,\n   4-6 impressions with exactly one decoy and 3-4 dated episodes each, 3-5 references, 2-4 arcs.',
    'decoy_phrase': 'exactly 1 impression is a decoy:',
    'episode_spread': 'two different weeks',
    'episodes_spec': '(3-4 episodes): `date` inside the horizon, spread over at least two different weeks',
    'events_bullet': '- `events`: 8-12 town events inside the horizon. `id` is `ev-<slug>`; `date` and `end_date` are ISO dates\n  (`end_date` null for a one-day event); `places` lists place ids. Include: the first day of the high-school year\n  and of the community-college fall term (both in the first week), the real holiday marked in the calendar, a\n  multi-day heat wave, something at the warehouse employer that touches workers (peak-season mandatory overtime, a\n  layoff rumor, a new scanner quota), something at the hospital (a staffing crunch, a union vote, a new charting\n  system), a local-politics item (a city-council candidate forum or a ballot measure), and 2-4 community events\n  (a street fair, a high-school football game, a farmers market, a library program). Each `description` says who\n  it touches and how.',
    'facts_range': '35-55',
    'first_use_spec': 'a date in the first two weeks of the horizon',
    'holiday_phrase': 'the holiday',
    'impression_changes': 'at most one impression changes.',
    'impressions_range': '4-6',
    'link_word': 'link',
    'places_range': '16-24',
    'principal_links': 'The two principals are linked to each other as the user message specifies.',
    'references_range': '3-5',
    'sensitive_classes': 'Exactly 1-2 `never_store` facts drawn from this list: a\n    government-ID or bank-account number, immigration status, a past arrest, a health diagnosis they keep private.\n    Exactly 1 `forgotten` fact: something mildly embarrassing or private they will tell the assistant and later ask\n    it to forget (a crush, an unflattering opinion of a boss, a debt to a friend).',
    'tier_bullet': "  - `tier`: `principal` (exactly the principals requested), `inner` (family, partners, close friends, bosses: 8-12\n    people across both principals, some shared, some only one principal's), or `periphery` (coworkers,\n    classmates, neighbors, a manager's manager: 10-16 people).",
}

PILOT = VillageSpec(
    name="pilot",
    start=date(2025, 8, 25),
    weeks=4,
    principals=(
        Principal((18, 22), "daily", "a community-college student at the town's community college who also works "
                  "part-time in town; the younger sibling of the other principal"),
        Principal((23, 30), "several_per_week", "a night-shift nurse at the town hospital; the older sibling of the "
                  "other principal"),
    ),
    links="They are siblings (link them as such).",
    required_links=((1, 2),),
    prompt_vars=PILOT_PROMPT_VARS,
    bible_bar=BibleBar(impressions=(4, 6), decoys=1, episodes_min=3, references_min=3, arcs=(2, 4), forgotten=(1, 1),
                       changes_min=2),
    events_min=6,
)

FULL_PROMPT_VARS = {
    "places_range": "24-36",
    "events_bullet": (
        "- `events`: 50-70 town events across the horizon, at least 3 in every month. `id` is `ev-<slug>`; `date` and\n"
        "  `end_date` are ISO dates (`end_date` null for a one-day event); `places` lists place ids. Include: the school\n"
        "  year (first days of the high-school year and of the community-college terms, finals weeks, winter and spring\n"
        "  breaks, prom, graduations), every real holiday marked in the calendar, the seasons (a late-summer heat wave,\n"
        "  wildfire smoke, winter storms, a spring allergy wave), recurring pressure at the warehouse employer (peak season\n"
        "  and mandatory overtime, a layoff round, a new scanner quota), the hospital (a staffing crunch, a union vote, a\n"
        "  new charting system), a local election cycle (candidate forums, a ballot measure, election day), and community\n"
        "  events through the year (street fairs, high-school games, the school play, farmers markets, library programs).\n"
        "  Each `description` says who it touches and how."
    ),
    "holiday_phrase": "the holidays",
    "tier_bullet": (
        "  - `tier`: `principal` (exactly the principals requested), `inner` (family, partners, close friends, bosses: 30-40\n"
        "    people across all principals, some shared, some only one principal's), or `periphery` (coworkers,\n"
        "    classmates, neighbors, a manager's manager: 45-55 people)."
    ),
    "principal_links": "The principals are linked to each other exactly as the user message specifies.",
    "cast_changes": "15-25",
    "cast_never_store": "2-6",
    "link_word": "links",
    "facts_range": "90-130",
    "changes": (
        "  - Changes: at least 24 facts change inside the horizon because of the arcs and the seasons (end the old fact on\n"
        "    the change date, set its `superseded_by` to the new fact, start the new fact that day). Spread change dates\n"
        "    across every month of the horizon, none in its last week, so later conversations can mention them."
    ),
    "sensitive_classes": (
        "Exactly 3-6 `never_store` facts drawn from this list: a\n"
        "    government-ID or bank-account number, immigration status (theirs or a family member's), a past arrest, a\n"
        "    health diagnosis they keep private, and for a minor their own age or date of birth.\n"
        "    Exactly 3-5 `forgotten` facts, spread across the year: things mildly embarrassing or private they will tell the\n"
        "    assistant and later ask it to forget (a crush, an unflattering opinion of a boss, a debt to a friend)."
    ),
    "impressions_range": "6-10",
    "episodes_spec": "(6-10 episodes): `date` inside the horizon, spread over at least three different months",
    "impression_changes": "at most two impressions change.",
    "decoy_phrase": "exactly 2 impressions are decoys, each",
    "references_range": "6-10",
    "first_use_spec": "a date spread across the first nine months of the horizon",
    "count_bar": (
        "1. Counts inside the ranges above: facts, at least 24 in-horizon changes, 3-6 never_store, 3-5 forgotten,\n"
        "   6-10 impressions with exactly two decoys and 6-10 dated episodes each, 6-10 references, 2-4 arcs."
    ),
    "episode_spread": "three different months",
}

FULL = VillageSpec(
    name="full",
    start=date(2025, 8, 25),
    weeks=52,
    principals=(
        Principal((15, 17), "several_per_week", "a high-school junior at the town high school applying to colleges, "
                  "with friction at home with their parent (principal 12)", minor=True),
        Principal((15, 17), "weekly", "a high-school sophomore with a first part-time job at a taqueria on the main "
                  "street, in the middle of friend-group drama; a student in principal 12's class and a cousin of "
                  "principal 6", minor=True),
        Principal((18, 22), "daily", "a community-college student at the town's community college who also works "
                  "part-time at a main-street cafe; the younger sibling of principal 7"),
        Principal((18, 22), "daily", "a junior at the nearby state-university campus who commutes by train and is "
                  "hunting for an internship; the roommate of principal 5"),
        Principal((18, 22), "several_per_week", "a sophomore at the state-university campus who works night shifts at "
                  "the warehouse employer; the roommate of principal 4"),
        Principal((18, 22), "weekly", "a community-college student in an automotive program who lives with family "
                  "and helps run the family food stand; a cousin of principal 2"),
        Principal((23, 30), "several_per_week", "a night-shift nurse at the town hospital; the older sibling of "
                  "principal 3 and a neighbor of principal 10"),
        Principal((23, 30), "daily", "a tech-support worker on the warehouse employer's IT desk; a coworker of "
                  "principal 5"),
        Principal((23, 30), "several_per_week", "a barista at the main-street cafe who is starting a small food "
                  "business; a coworker of principal 3"),
        Principal((30, 39), "daily", "a new parent on parental leave from a county office job, learning to run a "
                  "household with an infant; a neighbor of principal 7"),
        Principal((30, 39), "several_per_week", "a career changer leaving warehouse work for community-college "
                  "nursing prerequisites; in the same anatomy class as principal 3"),
        Principal((40, 49), "weekly", "a teacher at the town high school who is the parent of principal 1 and "
                  "teaches principal 2"),
    ),
    links="List the principals first, in this order. Link the principals exactly as their descriptions say.",
    required_links=((1, 12), (2, 12), (2, 6), (3, 7), (4, 5), (5, 8), (3, 9), (3, 11), (7, 10)),
    prompt_vars=FULL_PROMPT_VARS,
    bible_bar=BibleBar(impressions=(6, 10), decoys=2, episodes_min=6, references_min=6, arcs=(2, 4), forgotten=(3, 5),
                       changes_min=12),
    events_min=40,
    max_tokens={"r1_world": 32000, "r2_cast": 64000, "r3_bible": 64000},
)

FULL13_PROMPT_VARS = FULL_PROMPT_VARS | {
    "events_bullet": (
        "- `events`: 15-25 town events across the horizon, at least 3 in every month. `id` is `ev-<slug>`; `date` and\n"
        "  `end_date` are ISO dates (`end_date` null for a one-day event); `places` lists place ids. Include: the first days\n"
        "  of the high-school year and of the community-college fall term (both in the first week), midterms, every real\n"
        "  holiday marked in the calendar, the late-summer heat wave and the first fall storm, the warehouse employer's run-up\n"
        "  to peak season (hiring, mandatory overtime, a new scanner quota), something at the hospital (a staffing crunch, a\n"
        "  union vote, a new charting system), the local election in early November (candidate forums, a ballot measure,\n"
        "  election day), and community events (a street fair, high-school football games, the school play, Halloween,\n"
        "  a farmers market, a library program). Each `description` says who it touches and how."
    ),
    "cast_changes": "6-12",
    "facts_range": "50-80",
    "changes": (
        "  - Changes: at least 8 facts change inside the horizon because of the arcs (end the old fact on the change date,\n"
        "    set its `superseded_by` to the new fact, start the new fact that day). Spread change dates across every month\n"
        "    of the horizon, none in its last week, so later conversations can mention them."
    ),
    "sensitive_classes": (
        "Exactly 1-3 `never_store` facts drawn from this list: a\n"
        "    government-ID or bank-account number, immigration status (theirs or a family member's), a past arrest, a\n"
        "    health diagnosis they keep private, and for a minor their own age or date of birth.\n"
        "    Exactly 1-2 `forgotten` facts: something mildly embarrassing or private they will tell the assistant and later\n"
        "    ask it to forget (a crush, an unflattering opinion of a boss, a debt to a friend)."
    ),
    "impressions_range": "5-8",
    "episodes_spec": "(4-6 episodes): `date` inside the horizon, spread over at least two different months",
    "impression_changes": "at most one impression changes.",
    "decoy_phrase": "exactly 1 impression is a decoy:",
    "references_range": "4-6",
    "first_use_spec": "a date in the first two months of the horizon",
    "count_bar": (
        "1. Counts inside the ranges above: facts, at least 8 in-horizon changes, 1-3 never_store, 1-2 forgotten,\n"
        "   5-8 impressions with exactly one decoy and 4-6 dated episodes each, 4-6 references, 2-4 arcs."
    ),
    "episode_spread": "two different months",
}

# Shiran (2026-10-28): the dev village for now is FULL's 12 principals over its first 13 weeks, with the world, cast
# and bibles generated for that horizon only (dry-run option A).
FULL13 = VillageSpec(
    name="full13",
    start=date(2025, 8, 25),
    weeks=13,
    principals=FULL.principals,
    links=FULL.links,
    required_links=FULL.required_links,
    prompt_vars=FULL13_PROMPT_VARS,
    bible_bar=BibleBar(impressions=(5, 8), decoys=1, episodes_min=4, references_min=4, arcs=(2, 4), forgotten=(1, 2),
                       changes_min=6),
    events_min=15,
    max_tokens={"r1_world": 24000, "r2_cast": 64000, "r3_bible": 48000},
)

# D16: the held-out set is a second, smaller town from the same pipeline, frozen, used only to confirm winners. Same
# horizon and per-principal bars as full13 (so lag strata compare); the cast is sized for five principals.
HELDOUT = VillageSpec(
    name="heldout",
    start=FULL13.start,
    weeks=FULL13.weeks,
    principals=(
        Principal((15, 17), "several_per_week", "a high-school freshman in the marching band, finding a new friend "
                  "group; the child of principal 5 and a cousin of principal 2", minor=True),
        Principal((18, 22), "daily", "a senior in civil engineering at the state-university campus who drives for a "
                  "delivery app on weekends and is job hunting for after graduation; a cousin of principal 1 and the "
                  "roommate of principal 3"),
        Principal((23, 30), "daily", "a respiratory therapist on rotating shifts at the town hospital, planning a "
                  "wedding on a tight budget; the roommate of principal 2 and a neighbor of principal 4"),
        Principal((30, 39), "several_per_week", "a shift supervisor at the warehouse employer and single parent of two "
                  "young children, juggling childcare around peak-season hours; a neighbor of principal 3"),
        Principal((40, 49), "weekly", "the owner of a main-street hardware store, caring for an aging parent and "
                  "weighing whether to sell the business; the parent of principal 1"),
    ),
    links=FULL.links,
    required_links=((1, 5), (1, 2), (2, 3), (3, 4)),
    prompt_vars=FULL13_PROMPT_VARS | {
        "tier_bullet": (
            "  - `tier`: `principal` (exactly the principals requested), `inner` (family, partners, close friends, bosses: 12-18\n"
            "    people across all principals, some shared, some only one principal's), or `periphery` (coworkers,\n"
            "    classmates, neighbors, a manager's manager: 18-24 people)."
        ),
        "cast_changes": "3-6",
        "cast_never_store": "1-3",
    },
    bible_bar=FULL13.bible_bar,
    events_min=FULL13.events_min,
    max_tokens=FULL13.max_tokens,
    distinct_from="data/villages/full13-v1",  # its first world sample, with an identical prompt, was the dev town
)

VILLAGES = {s.name: s for s in (PILOT, FULL, FULL13, HELDOUT)}
