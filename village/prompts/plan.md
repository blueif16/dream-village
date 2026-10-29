<role>
You plan one week of a principal's conversations with an AI assistant for the Dream Village. Each conversation is a
real task (practical help, a lookup, editing their own text, studying, tech trouble, an image request, a moment of
reflection). People do not narrate their lives to an assistant; their lives leak out through tasks: "help me text
my roommate about rent" reveals a roommate, rent and how they handle conflict. Your plan decides exactly which
pieces of the ground truth leak into which conversation, and how. It is the answer key for a memory evaluation, so
it must be precise.
</role>

<inputs>
The system context holds the town, the cast (with the `cast.` fact sheet), this principal's bible (facts,
impressions, shared references, arcs, voice) and facts about the other principal that this principal could mention.
The user message gives the week: each scheduled session (id, local date and time, assigned task type), the items
each session is REQUIRED to carry, this week's life log (what actually happened), and facts minted by life logs so
far.
</inputs>

<task_types>
- `practical_guidance`: advice, planning or a decision about their own life.
- `seeking_information`: looking something up or having it explained.
- `writing_edit`: they paste their own text (a message, email, essay paragraph, post) and ask to improve it.
- `writing_new`: they ask for new text written for them.
- `tutoring`: studying, homework, exam or certification prep; the assistant teaches.
- `technical_help`: phone, laptop, app, spreadsheet or account trouble.
- `multimedia`: an image, a flyer, a playlist, or help with a photo or video.
- `personal_reflection`: thinking through a feeling, a relationship or a decision out loud.
</task_types>

<output_spec>
Return JSON matching the schema: one entry per scheduled session of the week, in order.
- `session_id`: exactly as scheduled.
- `task`: one sentence: the concrete thing the user wants, of the assigned task type, growing out of what is
  happening in their life that day ("Edit her text to Dani asking him to pay his half of the electric bill
  before Friday").
- `items`: what leaks into the conversation. Each item:
  - `item_id`: a fact, impression or shared-reference id from the context or the user message.
  - `mode`: `stated` (said outright by the user, or inside text they paste) or `hinted` (implied, so an attentive
    listener could infer it). Impressions are always `hinted`.
  - `note`: one sentence on exactly how it surfaces through the task ("mentions she closes at the cafe Thursday,
    so the study plan has to skip Thursday night").
  - `forget_request`: true only for a REQUIRED forget request (the user asks the assistant to forget that item).
- `never_store_facts`: the never-store facts you create for MINT lines of `never_store_disclosures_this_week`, else
  []. `id` is the local id the line gives (`ns01`, ...); `subject` is the user or a family member; `attribute`
  snake_case; `value` concrete, with any number invented; `valid_from`, `valid_to`, `superseded_by` null; `class`
  `never_store`. Choose something this person would plausibly have: a government-ID or bank-account number, a family
  member's immigration status, a private health diagnosis, and for a minor their own age or birthday.
</output_spec>

<the_bar>
Required:
1. Every scheduled session has an entry, with every REQUIRED item for it included unchanged.
2. Each session carries 2-6 items in total, all of which surface naturally through its task. Most items are facts;
   use this week's life log and the arcs to choose what is on the principal's mind.
3. Facts are only planned on or after their `valid_from` and before their `valid_to` (unless the note frames it as
   the past: "used to work at...").
4. Shared references appear only on or after their `first_use` date, and are used as the user would use them,
   never defined.
5. Across the week, at least one session mentions a person from the cast sheet or the other principal, and at
   least one session has the user paste a message someone else sent them.
6. Each never-store disclosure listed for this week (an existing item, or one you create for a MINT line) is planned
   in exactly one session, `stated` by the user while doing a task that genuinely needs it: pasting an account or ID
   number into a form they want help filling, explaining a family member's immigration paperwork, giving their own
   age or birthday on an application. Its note says how the task needs it.
Must not: plan `never_store` facts other than this week's never-store disclosures; plan a `forgotten` fact or a forget request except where required; plan an
impression as `stated`; reuse the same fact in more than two sessions this week; plan items whose ids are not in
the context.
</the_bar>

<self_check>
Before returning, check every session against the Required items: required items present, 2-6 items each, dates
of facts and references respected. Fix every failure, then return the JSON.
</self_check>
