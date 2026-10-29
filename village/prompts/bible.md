<role>
You are writing the character bible for one principal of the Dream Village: a resident whose AI-assistant
conversations will be generated and replayed through a memory system. The bible is ground truth. Facts are what a
good memory should keep current; impressions are what a perceptive assistant should come to understand without
ever being told; shared references are vocabulary that builds up between the user and the assistant. The
evaluation is only as good as this bible, so make it specific, internally consistent and usable by later writers.
</role>

<inputs>
The system context holds the town and the cast (people, links, and the shared fact sheet whose ids start with
`cast.`). The user message names the principal, their assistant usage, the horizon and its calendar, and the other
principal (whose bible is written separately).
</inputs>

<output_spec>
Return JSON matching the schema. Ids are local: `f01`..., `i01`..., `e1`... inside each impression, `r01`...,
`a01`... Refer to cast facts by their `cast.` ids.

- `summary`: 5-8 sentences: who they are at the start of the horizon, what they want, what is pressing on them.
- `facts` ($facts_range): the principal's own fact log, plus details only this principal would know about the people
  around them (a roommate's habits, a coworker's side hustle) that the cast sheet does not already cover.
  - `subject`: the principal's id or a cast person's id, never the other principal (their bible covers them).
  - `attribute`: snake_case, single-valued: one subject never holds two values of the same attribute at once, and
    never repeat an attribute the cast sheet already has for that subject.
  - Cover the ordinary texture an assistant would hear about: school or work and its schedule, housing and rent,
    money, transport, food and drink, phone and devices, hobbies, fitness, family logistics, plans and deadlines
    inside the horizon, small preferences.
  - `valid_from` / `valid_to` on [from, to); null `valid_from` for long-standing facts.
$changes
  - `class`: mostly `useful` (a later request would go better if the assistant remembered it) and `trivia` (true but
    nothing will need it; aim for about a third). $sensitive_classes
- `impressions` ($impressions_range): traits, values, patterns, relationship dynamics, recurring worries, or how they like to be
  spoken to. These are never stated by anyone; they are only shown.
  - `statement`: one sentence a perceptive friend would say ("Downplays her own achievements, especially in
    writing about herself").
  - `about`: the principal's id (for a relationship dynamic, still the principal: "Defers to her mother on money,
    then resents it").
  - `evidence` $episodes_spec;
    `description` is a concrete incident that shows the trait without naming it ("deletes the sentence about
    winning the regional science fair from her scholarship essay draft"). Each episode must be something that can
    surface in a task-driven conversation with an assistant.
  - `formed`: when it formed (null if long-standing). `changed` / `changed_to`: null and "" unless the trait
    shifts inside the horizon; $impression_changes
  - `decoy`: $decoy_phrase a plausible-sounding trait that is NOT true, which one misleading
    one-off episode seems to support ("skipped the gym all week" because she was sick). Its evidence holds that
    one-off episode plus 2-3 episodes that show the opposite is the norm. Its `statement` is the false trait.
  - `sensitive`: true for an impression a memory should not write down even as an inference (health, mental
    health, sexuality, finances in distress); at most one.
- `references` ($references_range): `term` the user will just say ("the Tuesday thing", "Grumpy Dave", "the spreadsheet"),
  `meaning` what it refers to, `first_use` $first_use_spec. Used, never defined.
- `arcs` (2-4): overlapping storylines across the horizon. `beats` are dated steps; `facts` lists the fact ids a
  beat creates, changes or touches. At least two arcs drive the fact changes above. One arc may touch a town event.
- `voice`: `texting_style` (capitalization, punctuation, emoji, typical length, typos, slang),
  `with_assistant` (how they phrase requests to an AI: how much context they give, politeness, whether they paste
  things, how they react to long answers), `quirks` (4-6), `sample_lines` (6-8 messages they might send an
  assistant, in their voice, about generic things rather than specific facts).
- `usage`: `frequency` exactly as given; `active_hours` 3-6 integers (0-23) when they usually open the assistant,
  fitting their schedule.
</output_spec>

<the_bar>
Required:
$count_bar
2. No two facts hold the same subject and attribute at the same time (including the cast sheet); every
   `superseded_by` and every beat fact id names a real fact.
3. Impression statements are never restated as facts, and their episodes show rather than tell.
4. The bible agrees with the cast sheet and the town: same names, jobs, places and event dates.
5. The person is specific: concrete values, a voice you could pick out of a group chat.
Must not: use real people, real places or brands; write facts about the other principal; describe sexual content,
self-harm or graphic violence; make the principal a minor.
</the_bar>

<self_check>
Before returning, walk through each Required item and mark it pass or fail; fix every failure, then return the JSON.
In particular, list every (subject, attribute) pair once to catch overlaps, and confirm each impression's episodes
have dates inside the horizon in at least $episode_spread.
</self_check>
