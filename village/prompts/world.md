<role>
You are the world-builder for the Dream Village, a simulated town whose residents use an AI assistant every day.
Everything you write becomes ground truth: later stages invent residents, their lives and their assistant
conversations on top of your town, and an evaluation scores a memory system against them. A vague or generic town
produces vague lives, so be specific the way a long-time resident would be.
</role>

<task>
Invent one fictional, mixed, mid-size town in the style of the San Francisco Bay Area, with the texture of about
80,000 residents, plus a calendar of town events for the horizon given in the user message.
</task>

<output_spec>
Return JSON matching the schema. Field meanings:
- `town`: the town's name. `region`: a short phrase placing it (an invented county or valley, "Bay Area-style").
- `summary`: 4-6 sentences: who lives here, what the town is like, its tensions.
- `texture`: 10-14 one-line, concrete details a resident knows (rent level and housing stock, commute and the
  invented transit line, the weather pattern in late summer, the mix of languages and families, what the
  warehouse jobs pay, where teenagers hang out, a running local argument).
- `places`: $places_range places. `id` is `pl-<slug>`; `kind` is a lowercase label. Six places MUST use exactly these kinds:
  `high_school`, `community_college`, `state_university` (a nearby campus of the state university system),
  `hospital`, `main_street`, `warehouse_employer` (a logistics or warehouse employer). Fill the rest with places
  ordinary lives pass through: apartment complexes, a grocery store, cafes, a taqueria or pho place, a gym, a park,
  a transit station, a church or community center, a laundromat, a library, a strip mall.
  `description`: 1-3 sentences that make the place specific (what it looks like, who goes there, a detail).
$events_bullet
</output_spec>

<the_bar>
Required:
1. Every name (town, county, streets, schools, businesses, the employer, the transit line, the hospital) is invented.
2. The six required place kinds are present exactly once each.
3. Events fall inside the horizon; their weekdays and $holiday_phrase agree with the calendar you were given.
4. At least four events plausibly cut across several kinds of lives (students, warehouse workers, hospital staff,
   families), because later stages tell one event from several residents' sides.
5. Details are concrete (a price, a street, an hour, a smell), not brochure adjectives.
Must not: name any real city, school, university system, employer, hospital, chain store, celebrity or public
figure (the state university is "<invented name> State", not a campus of a real system); generic references ("the
city across the bay", "the freeway", "the state university") are fine. Must not invent residents (a later stage does that).
</the_bar>

<self_check>
Before returning, check each Required item and fix anything that fails: scan every name for a real place or
brand, confirm the six required kinds, and confirm every event date against the calendar.
</self_check>
