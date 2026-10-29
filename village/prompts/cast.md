<role>
You are casting the Dream Village, a simulated town whose residents use an AI assistant every day. A memory system
will be scored on what it remembers about these people, so they must be specific, consistent and alive: people with
contradictions and histories, never types.
</role>

<inputs>
The system context holds the town (JSON). The user message names the principals to create: residents whose
assistant conversations will be generated. Everyone else you create exists because they touch a principal's life.
</inputs>

<task>
Create the principals exactly as specified, plus only the inner-circle and periphery people who touch them, the
links between all of them, and the shared fact sheet about the non-principals.
</task>

<output_spec>
Return JSON matching the schema.
- `people`: every person.
  - `id`: the lowercase first name (add `-<initial>` only to break a tie), unique.
$tier_bullet
  - `age`: an integer; principals inside their requested ranges.
  - `role`: a short phrase in town terms ("charge nurse on 4 West at <hospital>", "roommate, line cook at <cafe>").
  - `summary`: principals 3-5 sentences, inner 2-3, periphery 1. Name at least one contradiction for principals
    and inner-circle people (a frugal person who splurges on one thing, a confident nurse who dreads phone calls).
  - `traits`: principals 4-6, inner 3-5, periphery 1-2 short phrases.
- `links`: `{a, b, relation}` read as "a is b's <relation>" (`{"a": "rosa", "b": "kai", "relation": "mother"}`).
  $principal_links Every non-principal links to at
  least one principal. Add links between non-principals where they know each other.
- `facts`: the shared sheet, kept consistent for every principal who knows the person. Facts are ONLY about
  non-principals (a later stage writes each principal's own facts). Inner-circle people get 3-6 facts each,
  periphery 1-3.
  - `id`: `f01`, `f02`, ... `subject`: a person id. `attribute`: snake_case and single-valued, so one subject never
    holds two values of the same attribute at once (use `job`, `home`, `car`, `relationship_status`, `pet`,
    `shift_schedule`, `school_program`, `big_worry_this_fall`, ...; for a second value pick a distinct attribute).
  - `value`: concrete ("assistant manager at the Ferrell Ave laundromat since 2022").
  - `valid_from` / `valid_to`: ISO dates or null. A fact holds on [valid_from, valid_to): `valid_to` is the first day
    it no longer holds. Use null `valid_from` for long-standing facts.
  - `superseded_by`: when a value changes inside the horizon, end the old fact (`valid_to` = change date) and set
    its `superseded_by` to the new fact's id; the new fact starts that same day. Include $cast_changes such changes among
    inner-circle people (a parent's hours get cut at the warehouse, a roommate moves out, a boss leaves).
  - `class`: `useful` (likely to matter in later conversations), `trivia` (true, incidental), or `never_store`
    (sensitive: immigration status, a criminal record, a health diagnosis; use $cast_never_store).
</output_spec>

<the_bar>
Required:
1. Exactly the requested principals, with the requested ages, lives and $link_word.
2. Every non-principal links to a principal; nobody is there without a reason to show up in a principal's week.
3. Cast fact windows never overlap for the same subject and attribute, and every `superseded_by` names a real id.
4. People feel specific: names fit a mixed Bay Area-style town, ages and jobs fit the town you were given, and
   summaries carry concrete detail and a contradiction.
Must not: use real people or public figures; write facts whose subject is a principal; make anyone's age or
situation a minor's unless asked.
</the_bar>

<self_check>
Before returning, verify each Required item and fix any failure: count principals, check every link endpoint and
fact subject is an id in `people`, and scan facts for two values of one attribute held at the same time.
</self_check>
