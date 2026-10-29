<role>
You are writing one week of a principal's daily life log for the Dream Village. The log is ground truth: it is never
shown to the assistant directly; later stages draw the principal's assistant conversations from it. It records what
happened, not dialogue.
</role>

<inputs>
The system context holds the town, the cast (with the shared `cast.` fact sheet) and this principal's bible (facts,
impressions with dated evidence episodes, shared references, arcs, voice, usage). The user message gives the week:
its dates, the town events that overlap it, the arc beats, impression episodes and fact changes that fall inside it,
and the facts that start this week. You see only this week; other weeks are written separately from the same plan,
so stay faithful to the bible rather than inventing new storylines.
</inputs>

<output_spec>
Return JSON matching the schema.
- `entries`: 3-5 per day for every day of the week, in time order.
  - `date`: a day of this week. `what`: one or two sentences of what happened, concrete (place names from the
    town, people by name).
  - `with`: ids of the cast people involved ([] when alone).
  - `outcome`: how it ended or what it changed. `feeling`: the principal's feeling, in a few words.
  - `facts`: ids of bible, cast or new facts the entry shows, establishes or changes ([] if none).
  - `impressions`: ids of impressions the entry is evidence for ([] if none).
- `new_facts`: 3-8 small truths that arise this week and are not in the bible (a new phone case, a quiz score, a
  co-worker's new nickname, a plan for next weekend). `id` is `n01`, `n02`, ...; same field rules as bible facts;
  `valid_from` is the day it starts. Use attributes the subject does not already have. Mostly `trivia`, some
  `useful`. Never `never_store` or `forgotten`.
</output_spec>

<the_bar>
Required:
1. Every day has 3-5 entries, including the routine (shifts, classes, commutes, meals, sleep around a night shift).
2. Every arc beat, impression episode and fact change listed for this week appears in at least one entry on its
   date, tagged with the matching fact and impression ids.
3. Town events of the week show up where they touch this person (the heat wave changes the commute, the holiday
   changes the shift).
4. Consistent with the bible: no fact changes other than the listed ones; people, places and schedules match.
5. The decoy impression's one-off episode reads as a one-off with its real cause, not as a habit.
Must not: invent new major storylines; write dialogue; tag an impression on an entry that does not show it.
</the_bar>

<self_check>
Before returning, check every listed beat, episode and change against your entries (each must be present on its
date and tagged), and check each day has 3-5 entries. Fix gaps, then return the JSON.
</self_check>
