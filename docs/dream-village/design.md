# The Dream Village: design v0.1 (2026-10-28)

The Dream Village is the scorer for a self-optimization loop around our personal-memory system. It is a simulated town whose residents use an AI assistant every day. We generate their lives with a known truth, replay their assistant conversations through a memory system, ask questions at any point in time, and score the answers and the memory files. It is **not** a public benchmark.

**The system under test doesn't exist yet: we are inventing it.** Version 0 is the common core of the memory designs we researched (ChatGPT, Claude apps, Claude Code, Codex, Gemini and others): LLM-written markdown memory, navigated by the LLM, no embeddings. Where those designs disagree, the choice becomes a setting the loop can tune. It is unrelated to Memora and Memora-J (D20, D21). The optimization loop matters more than v0: we learn how the system works from its numbers.

## 0. Premise: the design has converged; the open problem is optimizing it (D40)
The major assistants with personal memory (ChatGPT, Claude apps, Claude Code, Codex, Gemini) have converged on the same design:
- a short, LLM-written profile always in context;
- an index of memory files, opened on demand;
- markdown memory the model writes and curates itself, a little after each conversation;
- periodic consolidation ("dreaming");
- a raw history kept as evidence;
- explicit rules for time and restraint.

They reached it independently, at production scale, for exactly our use case. So we take this design as the basis of the memory framework and do not try to reinvent it: it is the most efficient known shape for personal memory, and the burden of proof is on any departure (system-v0.md §1).

The question this project answers is **how to optimize on top of that design**: the specific details, settings and techniques that make it a robust memory layer in every area we score. Those areas are remembering facts at any age, understanding the person, restraint and forgetting, integrity of the files over time, and cost. Each candidate is one change to that common core, measured against the village (§1).

- **Decisions:** every decision below has an ID (D1, D2, …) that points to `decisions.md`, with its reasoning, its evidence, and when to revisit it. To change a decision, add a new row there; never edit an old one.
- **Evidence:** the research and measurements behind the decisions are in `evidence.md`.
- **Numbers:** every number here is a starting value (D6). They get tuned after we see results. The rules in §8 say how.

## 1. What the loop does
1. **Propose** a change to the memory system: a prompt, a timing, a model per step, or the memory format.
2. **Replay** the village's assistant sessions through the changed system.
3. **Score** it on the scorecard (§6).
4. **Keep** the change only if it passes the acceptance rule (§6.3).
5. **Log** every candidate, accepted or not, with its diff and its per-dimension deltas. Each candidate changes one thing, so effects can be attributed. This experiment log is a primary output: it is how we learn what makes a personal-memory system better (D19).

We score answer quality and cost only. We never test the mechanism itself, for example whether dreaming ran at 3 a.m. Consolidation timing is part of a system's configuration: the simulated clock triggers it, and we record when it ran (D2).

## 2. The village (D12–D16)
**The town:** a fictional, mixed, mid-size Bay Area-style city with the texture of about 80k residents. It has a high school, a community college, a nearby state-university campus, a hospital, a main street, and a logistics or warehouse employer. It follows the real calendar (holidays, seasons, school year), but its people and local events are invented, so models can't answer from training knowledge. Town events cut across several lives: a heat wave, a warehouse layoff, the school play, a local election.

**The cast comes in three tiers.** Only principals get full detail:

| Tier | Count | What is generated |
|---|---|---|
| Principals: people with an assistant whose memory we test | 12 | A daily life log, a character bible, a voice guide, assistant sessions |
| Inner circle: family, partners, close friends, bosses | ~35 | A dated sheet of key events and traits, kept consistent for every principal who knows them |
| Periphery: coworkers, classmates, neighbors | ~50 | Name, role, a few facts |

**Principals by age** (weighted toward the people who actually use assistants; see the evidence file, "Usage"):

| Age | Count | Example lives |
|---|---|---|
| 15–17 | 2 | High school, friend drama, college applications, a first job, friction with parents |
| 18–22 | 4 | Community college and the state university, working while studying, roommates, internships |
| 23–30 | 3 | A night-shift nurse, a tech-support worker, a barista starting a business |
| 30s | 2 | A new parent, a career changer |
| 40s | 1 | A teacher or small-business owner who is also a teen principal's parent |

**Rules for the cast:**
- Every principal is linked to at least one other principal (siblings, coworkers, teacher and student, parent and child), so shared events are told from several sides.
- Each principal has 2–4 overlapping arcs across the year, weekly routines, about a dozen recurring people, and a voice guide covering texting style and how they talk to an AI.
- Characters are specific people with contradictions, not types.
- **Minors:** the two teen principals' content stays age-appropriate.

**Horizon and usage:**
- 12 simulated months, starting at the beginning of a school year.
- Usage frequency: 4 principals use the assistant about daily, 5 several times a week, 3 weekly.
- That's roughly 2,000 sessions and 4–6M tokens.

**Held-out:** a second, smaller town with 4–6 principals, generated later by the same pipeline. It is used only to confirm winning changes (D16).

## 3. Ground truth (D4, D5, D13)
The world is planned top-down first, then rendered. We don't run a free simulation and read the truth off afterwards.

| Layer | Contents | How it reaches the assistant |
|---|---|---|
| **Fact log** | Subject (the principal or a named resident), attribute, value, `valid_from` / `valid_to`, `superseded_by`, class (`useful` / `trivia` / `never_store` / `forgotten`) | Stated or implied in sessions, according to the disclosure plan |
| **Character bible: impressions** | Traits, values, patterns, relationship dynamics, recurring worries, how the person likes to be spoken to; each with its evidence episodes and when it formed or changed | Never stated outright; only shown across several episodes |
| **Shared references** | Vocabulary that builds up between the user and the assistant ("the usual", "the landlord thing", nicknames) | Used, never defined |
| **Daily life log** (principals only) | Structured records of what happened, with whom, the outcome and the feeling. Not dialogue | Never directly; it is the source the disclosures are drawn from |
| **Disclosure plan** | For each session: which facts, impressions and references surface, and how (stated / hinted / never) | Drives rendering |

**Rules:**
- **The gold answer is what was disclosed by time t, not the whole life.** So the log can tell apart:
  - *told*: the assistant should know it;
  - *true but never told*: stating it is a fabrication or an over-inference;
  - *told but wrong*: the user's belief, which must be remembered as their belief.
- **Decoys:** plant one-off events that must not become traits ("skipped the gym because she was sick").
- **Recoverability check:** an independent model, from a different model than the generator, reads only a trait's evidence episodes and must infer the trait blind. If it can't, regenerate those episodes or tag the trait `unrecoverable`.
- **Keep or forget is decided at construction:**
  - `useful`: some later probe needs it, meaning the probe succeeds with the history and fails without it;
  - `trivia`: nothing ever needs it;
  - `never_store`: set by sensitivity norms seeded from published taxonomies;
  - `forgotten`: the user asked.
- **Code validator, no LLM.** It checks that:
  - validity windows for the same attribute never overlap;
  - every change is disclosed after it happens;
  - every needed fact is disclosed before the probe that needs it;
  - every forget request comes before the probes that test it.

## 4. What gets rendered (D13)
- **Only principal ↔ assistant sessions are rendered as text.** Conversations between people are ground truth and are never rendered.
- **Sessions follow real usage:** practical guidance ~29%, seeking information ~24%, writing ~24% (mostly editing the user's own text), tutoring ~10%, technical help ~5%, personal reflection ~2%.
- **Life is revealed incidentally through tasks** ("help me text my roommate about rent"), not by narrating the day. Principals occasionally paste a message from someone else ("my sister sent this").
- **Verification pass:** extract every claim a session makes about the user or the people in their life, and compare it with the log.
  - A planned item is missing → regenerate the session.
  - An unplanned item appears → add it as `trivia`, or regenerate if it conflicts.
  - Track the pass rate.
- **Model family:** about 15% of principals are generated with a non-Claude model, as a check on same-family bias (D11).

## 5. Timeline, snapshots and probes (D3, D17)
**Simulated clock:** every session carries a timestamp. The harness calls the system's end-of-session hook, and its own consolidation trigger fires on the simulated clock.

**Snapshots:** a git commit whenever the state changes (after a write, at a session end, after a consolidation). A snapshot holds:
- the external memory layers;
- what is loaded into the live context at that moment (the current session so far, plus anything injected at session start);
- the config and model versions.

The memory is markdown, so it lives in git directly. If a setting ever adds an index, it is rebuilt from the text at checkout, deterministically.

**Probes are read-only forks.** A question at time t runs on a copy of the snapshot at t, and nothing it writes goes back into the timeline. Behaviours that unfold over a conversation, like repeatedly bringing up the same memory, need a short scripted multi-turn exchange on the fork.

**Probe kinds (starting mix):**

| Kind | Share | Example |
|---|---|---|
| Ambiguous request that depends on an impression | ~40% | "Help me write a LinkedIn post about my new role" (she downplays her achievements) |
| Fact-based request | ~25% | Dinner suggestions that must respect "vegetarian since March" |
| Resolving a shared reference | ~15% | "Any update on the landlord thing?" |
| Restraint: no memory applies, or a near-miss distractor | ~20% | A request where a similar-looking memory must stay out |

Time effects cut across every kind: staleness, dates passing (a plan becomes past, and passing doesn't mean it happened), traits changing, recurrence (birthdays). The lag between disclosure and question is recorded as an analysis dimension: same session, same day, after a consolidation, after the fact leaves the recent window, weeks or months later.

**Every probe stores, from construction:**
- the request and time t;
- per-item targets (`must_use` / `may_use` / `must_not` / `stale`) over facts, impressions and references;
- the answer-shaping criteria for impressions;
- its kind and lag.

**Shortcut filter:** run each probe with no memory, and drop or flag any memory-needing probe that still passes.

## 6. Scoring (D6–D9)
### 6.1 The scorecard
| Dimension | What it measures | Measured by |
|---|---|---|
| **Remembering** | Facts and shared references used correctly and currently, at any lag | Answer checks on fact and reference items |
| **Understanding** | Impressions shaping ambiguous requests, without over-inference | Answer checks on impression items |
| **Restraint** | Nothing irrelevant, stale or forgotten intrudes | `must_not` and `stale` checks, restraint probes |
| **Integrity** | The memory files stay correct over the timeline | Per-commit store audit (§6.2) |
| **Cost** | Write-side $ per simulated day, context tokens per answer, memory size | Ledger |

**Answer checks are atomic yes/no questions**, judged one answer at a time against the probe's references:
- Is each `must_use` item used correctly?
- Is each `must_not` item absent?
- Is each `stale` item not presented as current?
- Does the answer follow the impression criteria without labeling the person?
- Is the task done?

`may_use` items aren't scored. Use code instead of an LLM wherever possible: a forbidden value is a string or ID match. Each probe also gets a no-memory answer and an oracle answer (given the full bible), so **memory gain captured** = (system − no memory) / (oracle − no memory) can be reported.

### 6.2 The integrity audit, per commit
1. Extract the verifiable claims from the memory files.
2. An LLM maps each claim to a fact, impression or reference entry, or to "none".
3. Code labels each claim from the dates and classes:
   - `supported`;
   - `stale` (the entry has ended);
   - `contradicted`;
   - `fabricated` (no entry);
   - `over-inferred` (no trait supports it, or it came from a decoy);
   - `premature` (fewer evidence episodes than the trait needs);
   - `leaked` (`never_store` or `forgotten`).
4. A `useful` entry valid at t with no supporting claim counts as `missed`.

Only new or edited claims (from the git diff) need the LLM; status changes over time are computed by code. Report precision, stale-shown-as-current rate, fabrication and over-inference rates, leak count, recall of useful entries, and size, each plotted along the timeline.

### 6.3 Acceptance
- **Hard gates (any failure means reject):**
  - no leak of `never_store` or `forgotten` items;
  - fabricated, stale-shown-as-current and over-inferred claims don't rise above the current version at any checkpoint;
  - memory stays under its size cap;
  - output contracts pass their deterministic checks.
- **Targeted improvement:**
  1. Pick a weak dimension and try changes aimed at it.
  2. Screen each change on a small set: 3 principals over 3 months.
  3. If it passes, run the full dev village.
  4. Accept only if the target dimension improves by more than the noise (measured by re-running the current version against itself), and **no other dimension drops beyond that noise.**
  5. Confirm on the held-out town before landing.

  Improving all dimensions at once uses the same rule with no target.

## 7. Judges and human validation (D9, D11, D18)
- **Model roles:**

  | Role | Model |
  |---|---|
  | Generator | A strong model, with ~15% of principals from another vendor |
  | Memory curator (what we optimize) | Cheap; the model choice is one of the loop's settings |
  | Answerer | Fixed within an experiment |
  | Recoverability check | A model different from the generator |
  | Primary judge | A strong model from a different family than the curator and answerer |
  | Second judge | A strong Claude model |

  Self-preference bias follows model family, not model size.
- **Veto:** on `must_not` and `stale` checks, an item fails if either judge flags it. Judges are chosen by their true negative rate (TNR), the rate at which they catch failures, on human-labeled items, not by overall agreement.
- **Pinning:** store the judge snapshot, prompt hash and rubric version with every score. Re-judge 10% of items each run to track flips.
- **Humans validate the generator and the judges. They don't label preferences.** Labels are click-only context cards: the request, the response, and the one entry (value, dates, disclosure excerpt, previous value), answered yes or no. Nobody reads the history.
  - Start with the three riskiest check types (`must_not`, `stale`, `fabricated`): about 100 labels each, with at least 50 passes and 50 failures.
  - Impression checks get the next batch, with two raters each.
  - Accept a judge when its true positive rate (TPR, the rate at which it passes correct items) and its TNR are both ≥ 0.9 and κ ≥ 0.8.

## 8. Changing the eval (D10)
- The probe mix, weights, rubric and judge each carry a version. After any change, re-score the current version of the system on the new eval version before comparing anything. Scores are never compared across eval versions.
- Eval changes come only from failure analysis on dev principals, and are logged in `decisions.md` with the evidence that prompted them.
- The held-out town is frozen. It confirms winners and never shapes the eval.

## 9. Sizing formulas (recompute when anything changes)
- **Checks needed**, for paired yes/no items, α = 0.05, 80% power: n ≈ 7.84 · d / δ², where d is the share of items that change outcome between versions and δ the gain to detect. At d = 0.10: about 900 checks for 3 points, 2,000 for 2 points, 7,800 for 1 point. Multiply by about 1.56 for judge error at TPR = TNR = 0.9. Measure d on the first baseline-vs-baseline run.
- **Probes per principal:** with about 5 checks per probe, 12 principals × 40–50 probes gives about 2,500 checks.
- **Replay cost per candidate:** history tokens × the system's write cost per million tokens. That rate is unknown for our harness. Track A's Memora build rate (~$5–6/M, so ~$25–35 for the full dev village and ~$3 for the screening set) is only a placeholder until the pilot measures it.
- **To grow the village:** if per-dimension CIs are wider than the gains we're chasing, add principals (better) or probes per principal (cheaper, but they're correlated within a principal). Log it as a decision.

## 10. Build order
1. Generator and code validator. Pilot: 2 linked principals × 1 month, checked by hand.
2. System v0 (O3), then timeline replay with git snapshots, plus baselines: no memory, full context, recent window, oracle.
3. Probes, judges and the first context-card labeling pass.
4. A baseline run plus a baseline-vs-baseline run to measure noise. This sets d and the real cost per replay.
5. First optimization rounds.

## 11. Open items
| # | Item | Blocks |
|---|---|---|
| O1 | Who labels: Shiran alone, or with 1–2 teammates for agreement | Step 3 |
| O2 | A non-Claude API key for the primary judge and the cross-family slice of generation | Cross-family pieces of §4 and §7 |
| O3 | Build system v0 as specified in `system-v0.md` (D22, D23): config plus prompt files, so each candidate change is one diff | Step 2 |
| O4 | Claude credits expire about 2026-10-30, so all of this runs at normal API prices. Measure the real replay cost on the pilot before sizing | Step 4 |
| O5 | System changes suggested by the research (an append-only raw log as evidence, small edits applied by code instead of whole-file rewrites, a size cap rather than a brevity reward). These are candidates for the first optimization rounds, not eval decisions | Step 5 |
| O6 | ~~Inference policy~~: resolved by D24 (tagged `[inferred]` lines allowed for non-sensitive patterns) | — |
