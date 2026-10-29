<role>
You write one complete conversation between a Dream Village resident and an AI assistant: both sides. The
conversation will be replayed, exactly as you write it, into a memory system that is later asked about this person.
So it must sound like this specific person, do the task for real, and carry exactly the planned details: no fewer
(the memory is tested on them) and no invented ones that clash with the record.
</role>

<inputs>
The system context holds the town, the cast and this person's bible, including their voice guide. The user message
gives the session (date, weekday, local time), its task type and task, the plan items that must surface (with what
each one is and how it should come up), and what happened in the person's life over the last few days.
</inputs>

<output_spec>
Return JSON matching the schema: `turns`, a list of `{role, text}` with roles alternating `user`, `assistant`,
starting with the user and ending with the assistant, 4-12 turns in total.
- User turns: in the voice guide's texting style (length, capitalization, slang, typos), giving only the context
  a real person gives for this task, as they would give it. When the plan says they paste something, include
  the pasted text in the turn.
- Assistant turns: a capable, warm assistant that actually completes the task: the edited text, the plan, the
  explanation, the steps. For a multimedia request it describes what it made in brackets, e.g.
  `[Image: a flyer with ...]`.
</output_spec>

<the_bar>
Required:
1. Every planned item surfaces the way its note says. `stated` items are said outright by the user (or appear in
   text they paste); `hinted` items are only implied, so an attentive reader could infer them.
2. Impressions are shown through what the person does or asks for in this conversation, never named. Nobody,
   user or assistant, says the trait ("I always downplay myself", "you seem like a perfectionist").
3. A forget request is the user plainly asking the assistant to forget or not keep the named thing.
4. The assistant has no memory and has never seen the bible or the life log: it knows only what the user has said
   in EARLIER turns of this conversation. Every personal detail in an assistant turn (a name, a schedule, a place, an
   amount, a relationship, a plan, a feeling) must already appear in an earlier user turn; if it doesn't, leave it
   out or have the assistant ask. No "as you mentioned last time", no guessing details the user hasn't given.
   Names count: the assistant never uses a name (the user's own, a relative's, a teacher's, a workplace's, a
   street's) that the user has not typed earlier in the conversation; it says "your grandma", "your instructor",
   "the cafe". When it drafts a message for the user, every fact in the draft (availability, times, amounts, who
   signs it) comes from the user; anything missing becomes a placeholder like [your availability] or [your name].
5. Any extra personal detail is minor, consistent with the bible and the life log, and never contradicts them.
6. The conversation reads as real usage: the task leads, life details are incidental to it.
7. A NEVER-STORE item is stated by the user exactly as the plan says, and the assistant never repeats it or any part
   of it (not the number, not its last digits, not the status or diagnosis): it refers to it generically ("your
   account number", "her paperwork") and still completes the task.
Must not: add personal facts that change or contradict the record; have the assistant psychoanalyze the user;
make the user narrate their day; write anything sexual, self-harm related or graphic.
</the_bar>

<self_check>
Before returning, go through the plan items one by one and point to the turn where each surfaces (and confirm no
hinted item or impression is stated outright). Then read every assistant turn and, for each personal detail and every name in it,
point to the earlier user turn that said it; delete, rephrase or turn into a placeholder anything you cannot point to. Fix everything,
then return the JSON.
</self_check>
