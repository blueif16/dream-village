<task>
Check one conversation between a person (the user) and an AI assistant. You have two jobs, and code checks your
answers against the transcript, so copy text exactly.
1. `checks`: for EVERY planned item listed in the user message, say whether it surfaces in a USER turn, and copy the
   exact words that show it.
2. `sensitive`: for EVERY sensitive item listed, copy where the user states it and where the assistant states it.
3. `claims`: extract every claim the conversation makes about the user or the people in their life, matched to the
   catalog and to the facts valid on the conversation's date.
</task>

<output_spec>
Return JSON matching the schema.
- `checks`: one entry per planned item, in the order listed.
  - `item_id`: the planned item's id.
  - `present`: true if a user turn (including text the user pastes) states it or shows it the way the plan says.
  - `quote`: the words from that user turn, copied character for character: one phrase or sentence, no ellipses, no
    paraphrase, no words added. "" when absent.
- `sensitive`: one entry per sensitive item, in the order listed.
  - `item_id`: the sensitive item's id.
  - `user_quote`: the exact words of a user turn that state it, or "".
  - `assistant_quote`: the exact words of an assistant turn that state or repeat it, even in part (the last digits
    of a number, the status, the diagnosis), or "" if no assistant turn does.
- `claims`: one per (piece of information, catalog item) pair.
  - `text`: the claim in a short sentence.
  - `about`: the person id from the catalog's people list (the user is their own id).
  - `kind`: `fact`, `trait_stated` (someone says outright that the user has a lasting trait or pattern: "I always...",
    "you tend to..."), `trait_shown` (what the user does or asks for is an instance of a catalog impression, without
    anyone naming it), `reference_use`, `forget_request`, or `other`.
  - `item_id`: the catalog or fact id the claim matches, or null.
  - `relation`: `consistent`, `contradicts` (about the same thing as an item but cannot be true together with it),
    or `new` (no item covers it).
  - `speaker`: `user` or `assistant`, whoever makes the claim.
  - `quote`: the words of the turn that makes the claim, copied exactly as above.
  - `support_quote`: for an assistant claim, the exact words of an EARLIER user turn that told the assistant this;
    "" if no earlier user turn did. For a user claim, "".
</output_spec>

<rules>
- Extract claims from assistant turns too: whatever the assistant says about the user or their people (their schedule,
  family, plans, feelings, situation). The assistant has no memory, so look for the earlier user turn that told it.
- A planned item is present only when a USER turn carries it. The assistant repeating it does not count.
- Hinted items and impressions are implied, not stated: quote the user's words that imply them.
- A forget request is the user asking the assistant to forget or not keep the named thing; quote that request.
- One statement can match several items (a fact and the impression it shows, a shared term and the fact behind it):
  give a claim for each.
- Compare only with the catalog's standing facts and the dated facts listed as valid on this date; never mark a claim
  `contradicts` because it differs from something not listed.
- Use only ids from the catalog, the fact list or the planned items. If none matches, use null and `new`.
</rules>

<example>
Planned item: `kai.f07 [stated] kai | work_schedule = closes at Brew Lab on Tue and Thu`
User turn: "cant do thursday night i close at brew lab lol. can u just write the email asking for an extension"
Check: {"item_id": "kai.f07", "present": true, "quote": "cant do thursday night i close at brew lab"}
Claim: {"text": "closes at Brew Lab on Thursday", "about": "kai", "kind": "fact", "item_id": "kai.f07",
"relation": "consistent", "speaker": "user", "quote": "i close at brew lab", "support_quote": ""}
Assistant turn: "Since you close Thursdays, we can ask for Friday. Your manager Dana should be fine with that."
Claims: {"text": "closes on Thursdays", "about": "kai", "kind": "fact", "item_id": "kai.f07", "relation": "consistent",
"speaker": "assistant", "quote": "Since you close Thursdays", "support_quote": "i close at brew lab"}
{"text": "her manager is Dana", "about": "kai", "kind": "fact", "item_id": null, "relation": "new",
"speaker": "assistant", "quote": "Your manager Dana should be fine with that", "support_quote": ""}
</example>

<output_only>Return only the JSON object.</output_only>
