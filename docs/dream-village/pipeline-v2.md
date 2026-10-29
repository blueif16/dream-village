# Dream Village pipeline v2: who sees what (draft for Shiran, 2026-10-29)

Status: **proposal, not built.** It replaces how sessions are written (design.md §4, D27) and how towns are made
independent; the eval (§5–§7) is unchanged except where noted.

## 1. Why v1 broke
- **One call wrote both sides of every conversation.** D26 (one Batch round per stage, to save latency) and D27 (fixed
  scripts, "the generator writes both the user's and the assistant's turns") left no room for a turn-by-turn
  exchange, so the render call got the town, the full cast and the person's whole bible and wrote the user *and* the
  assistant. The rule "the assistant knows only what the user said" lived only in an instruction (render.md rule
  4), then a checker (D32 d), a name filter (D35 d) and regenerations. Patches on a leaking structure.
- **The cost of the patches:** the checker is 40% of generation cost ($8.17 of $20.29 for heldout-v1); 19–23% of first
  drafts fail it; "ungrounded" is 72–78% of its flags, most of them false alarms in a hand sample (advice like "pack
  your thermos", or "9:00" after the user said "at 9"); regenerations add new flags (30 in full13).
- **Every town came from one fixed world prompt,** so a second town came back as the first (heldout-v1 attempt 1).

## 2. The invariant v2 adds
**Every step is a separate agent with a written input list. Code builds each request from that list and nothing
else, and a test asserts it** (for example: the assistant agent's request contains no bible, plan or fact text). A
boundary that only an instruction enforces doesn't count as a boundary.

## 3. The steps
| # | Step | Agent | Sees (and nothing else) | Writes |
|---|---|---|---|---|
| 0 | Town setting | code + Opus | parameters drawn per town by code (region type, size, economy, employers, demographics, climate) | a setting card; every town an independent thread |
| 1 | World | Opus | its setting card, the calendar | town, places, events |
| 2 | Cast | Opus | the world, the principal specs | principals, inner circle, periphery, links, shared facts with who knows each |
| 3 | Bible, per principal | Opus | world, cast | facts (class, validity), impressions (traits with evidence episodes, decoys), shared references, arcs, voice guide |
| 4 | Life log, per principal-week | Sonnet | world, cast, that bible, the arc plan | what happened, with whom, outcome, feeling, new facts |
| 5 | Disclosure plan, per principal-week | Sonnet (the director: sees all) | bible, life logs so far, schedule | each session's task and its items (stated / hinted / forget / never-store), **at most N items per session** |
| 6 | Validator | code | all of the above | violations (overlaps, order of disclosure, forget before probe) |
| 7a | **User turns** | Sonnet, one thread per session | **only this person's view:** their bible and voice, recent life log, the people they know and what they know about them, this session's task and items, the conversation so far | the next user turn, or END |
| 7b | **Assistant turns** | a separate thread per session | **only the date/time and the conversation so far.** No bible, no plan, no cast, no memory | the next reply |
| 8 | Disclosure check | Haiku if it matches Sonnet on a 50-session sample, else Sonnet | transcript, the session's items, facts valid that day | per item: a verbatim quote from a user turn (code confirms it), "stated outright" for traits, contradictions in user turns |
| 9 | Recoverability | Sonnet (not the generator) | only a trait's evidence sessions | the trait, inferred blind |
| 10 | Gate | code | 6, 8, 9 | pass / fail per town |

Turns 7a/7b run in rounds: round k carries turn k of **every** session of every town at once, so a whole batch of
towns needs ~12 rounds, not 12 per session (D26's latency rule still holds). Each person's persona prefix is
prompt-cached within their own thread. Scripts stay fixed after generation, so replays stay deterministic (D27's
reason survives; its "one writer" doesn't).

**Gold comes from step 8, not from the plan:** an item counts as disclosed only where code confirmed the user's quote.

## 4. What each check checks
- **Generation (steps 6, 8, 9):** the truth is consistent (code); every planned item really reached the transcript,
  in a user turn (quote + code); traits are shown, never named; user turns don't contradict the record; each trait
  is recoverable from its evidence alone. "Ungrounded assistant" checks disappear: the assistant can't know anything
  the user didn't type.
- **Evaluation (unchanged, design §5–§7)**, the agreed matrix:
  - *Remembering:* facts and shared references used correctly and currently, at every lag.
  - *Understanding:* impressions (deeper traits, preferences, how they like to be helped) shaping ambiguous
    requests, without labeling the person or over-inferring from decoys.
  - *Restraint:* nothing irrelevant, stale, forgotten or never-store intrudes.
  - *Integrity:* the memory files stay true to what was disclosed (per-commit audit).
  - *Cost:* write $ per simulated day, context tokens per answer, memory size.
  - Judged per check, yes/no: Opus primary, Sonnet second, code where possible; validated against human labels.

## 5. The self-optimization loop
Intended (design §1): propose one change → replay → score → keep only past the noise with no drop → log every
candidate. Today: one round, run by hand (5 candidates, nothing landed, D42); no automatic proposer, no archive.

## 6. Other rigor gaps (each needs a fix before results are claimed)
1. **Understanding is underweighted:** eval v2 has 69 impression probes of 443 (16%); design §5 planned ~40%.
2. **Judges are Claude-only and unvalidated:** no human labels yet (the labeling page exists; §7 needs ~100 per
   risky check type). All judge-based numbers are provisional.
3. **Noise floor** comes from one A/A pair on eval v1; re-run A/A on the current eval.
4. **Screen deltas** in scorecard.md are unpaired (candidate on 4 principals vs v0 on 12); the scorer should report the
   paired delta.
5. **No non-Claude generation** since D28 (design §4 planned ~15%), so same-family bias is unchecked.
6. **13 weeks, not 12 months:** no yearly recurrence; long-lag probes are thin.
7. **Sample size:** 12 dev principals resolve ~4-point differences.

## 7. Open choices
- Assistant-agent model: Sonnet (a realistic assistant; short contexts keep it cheap) or Haiku.
- Turn rounds through Batch (~12 rounds) or real time (faster, no Batch discount).
- Keep off-policy scripts (the session assistant is memoryless; the tested system only acts at probe time), or let
  the tested system play the assistant (on-policy, D27's revisit trigger).
