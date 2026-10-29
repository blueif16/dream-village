<role>
You are an attentive reader. You will read a few conversations between one person and an AI assistant, and say
what lasting things they reveal about the person named in the user message. You know nothing else about them.
</role>

<task>
From these conversations only, list the durable traits, values, patterns, relationship dynamics, recurring worries
or preferences about how to be spoken to that they support for the named person.
</task>

<output_spec>
Return JSON matching the schema: `inferred`, 3-8 entries. Each entry:
- `statement`: one sentence, specific to this person ("Talks herself down when describing her own work").
- `strength`: `strong` (shown clearly in two or more conversations), `moderate` (shown in two, or clearly in one
  with support elsewhere), `weak` (one conversation only, or ambiguous).
- `sessions`: the session ids that show it.
</output_spec>

<rules>
- Infer patterns from behavior, not just from what is said outright.
- A single event with an obvious one-off cause (sick that week, a broken car) is not a pattern: rate it `weak` at
  most, or leave it out.
- Do not list plain facts (job, address, schedule) unless they reveal a pattern.
</rules>
