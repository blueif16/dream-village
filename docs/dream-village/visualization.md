# Dream Village Observatory: visualization design brief (v0.1, 2026-10-28)

**For:** a design or build agent making a web-based visualization of the Dream Village. Read all of it before designing. The spec lives in `memo-bench/docs/dream-village/` (design.md, decisions.md, system-v0.md, evidence.md); this brief is self-contained but defers to those files.

## 1. What you are visualizing (read this first)
The Dream Village is a **simulated town whose residents talk to an AI assistant every day.** We generate their lives with a known truth and feed their assistant conversations to a memory system. Then we ask the assistant questions at any moment in time, and score:
- **the answers:** did the right memories shape them, and did the wrong ones stay out?
- **the memory itself:** is it correct, current and uncorrupted?

A self-optimization loop keeps changing the memory system, one change at a time, and keeps only changes that improve the score without hurting anything else. **The point of the whole project is to watch a personal-memory system evolve, and to learn what makes one better.**

The visualization must make one idea felt, not just shown:

> **There is the town as it is, and the town as each assistant remembers it. The gap between them is what we measure, and the loop is how the gap closes.**

Everything below serves that sentence. A view that doesn't serve it doesn't belong.

## 2. Concepts you must represent (glossary)
| Concept | Meaning | Where it comes from |
|---|---|---|
| **Principal** | One of 12 residents who has an assistant whose memory we test. Ages mostly 15–30, a few in their 30s and 40s | cast |
| **Inner circle / periphery** | ~35 / ~50 other residents: family, friends, coworkers, neighbors. They have real, consistent lives but no assistant | cast |
| **Truth: fact log** | Dated facts about a person (value, `valid_from`, `valid_to`, `superseded_by`), each with a class: `useful` / `trivia` / `never_store` / `forgotten` | principal bible |
| **Truth: impressions** | Traits, patterns, values and relationship dynamics that are **never stated outright**, only shown across several episodes. Some are **decoys**: one-off events that must NOT become traits | principal bible |
| **Truth: shared references** | Vocabulary built up between the user and the assistant ("the landlord thing", "the usual") | principal bible |
| **Daily life log** | What happened to a principal each day, with whom, the outcome and how they felt. The assistant never sees this directly | life log |
| **Disclosure** | Which slice of life the principal brings to the assistant, and how: `stated` or `hinted`. Most of life is never told | disclosure plan |
| **Session** | One conversation between a principal and the assistant, rendered as text. These are fixed scripts. **This is the only thing the memory system ever sees** | sessions |
| **Memory files** | The assistant's markdown memory, in Claude's layout: `/profile.md` (who they are, ≤300 words), `/preferences.md` (how they want the assistant to behave), `/topics/<domain>.md` (habits, tastes), `/areas/<name>.md` (ongoing involvements), `/people/<name>.md` (relationship context). Each line is tagged `[stated]` or `[inferred]`; inferred lines cite their evidence episodes | system snapshots |
| **Fast write** | After each turn, a background pass files what is durable | system snapshots |
| **Dream** | Nightly consolidation: merge duplicates, resolve contradictions, turn "next Saturday" into a date, mark past plans, prune | system snapshots |
| **Snapshot / commit** | The full memory state after every change, kept in git. Time travel = checking out a commit | system snapshots |
| **Probe** | A task request asked at a moment t (e.g. "help me write a LinkedIn post about my new role"), answered on a **read-only copy** of memory at t | probes |
| **Checks** | Atomic yes/no judgments per probe: was each `must_use` item used? is each `must_not` item absent? is each `stale` item not presented as current? did the answer follow the impression without labeling the person? is the task done? | probe results |
| **Integrity labels** | Each memory line is `supported`, `stale`, `contradicted`, `fabricated`, `over-inferred` (from a decoy, or unsupported), `premature` (too little evidence), or `leaked` (never-store or forgotten). A useful truth with no memory line is `missed` | integrity audit |
| **Scorecard** | Five dimensions: **Remembering** (facts and references), **Understanding** (impressions), **Restraint** (nothing irrelevant, stale or forgotten intrudes), **Integrity** (memory correctness over time), **Cost** | scorecard |
| **Lag** | How long ago the fact was disclosed: same session · same day · after a dream · outside the recent window · weeks or months | probes |
| **Memory gain captured** | (system − no memory) / (oracle − no memory), per probe and overall | probe results |
| **System version / candidate** | Each candidate is ONE change (a prompt, a timing, a model, a format setting). It is accepted or rejected; every candidate is logged with its diff and per-dimension deltas | candidate log |

## 3. The core visual idea: two towns, superimposed
- **The Real Town (truth):** solid, warm and fully lit, rendered from the truth files.
- **The Remembered Town:** each assistant's understanding of its principal, drawn as **ink on glass**. A translucent sketch layered over the real thing that gets drawn in, stroke by stroke, as memory lines are filed.

**The integrity labels become visual states of the ink:**

| Label | Look |
|---|---|
| supported | ink sits exactly on the real object |
| stale | ink yellowed like an old photo, offset from where the object has moved |
| fabricated | an ink ghost with nothing real underneath |
| over-inferred | ink that extends a shape beyond its truth, dashed |
| leaked | a redaction bar where nothing should be |
| missed | a real object with no ink on it at all: a hole in the memory |

Never rely on color alone: use line style, fill and texture as well.

**The knowing funnel** is the same idea for each principal: **Lived → Told → Remembered → Used when relevant.** It is the single most important number-picture in the product. It should be visible in some form in every person-level view.

## 4. Views
Each view must answer one named question. Build them in the priority order shown.

### 4.1 Person view: "What does the assistant know about Dana, and how did it come to know it?" (P1)
Three bands for one principal over time:
- **Lived:** the life log as a timeline of events.
- **Told:** the sessions, as chat bubbles placed on the same timeline. Each disclosure is a thread down from the life event it came from. Most events have no thread, which is the point.
- **Remembered:** the memory files as cards grouped by Claude's folders. Each line carries its integrity glyph.

**Evidence threads:** hovering an `[inferred]` line lights up the session moments that taught it, and the bible impression it should match. A decoy-born inference shows its thread in the over-inferred style.

**The person as a constellation:**
- `/profile` at the center;
- `/people` as figures around it;
- `/areas` as orbiting places;
- `/topics` as personal objects;
- `/preferences` as a "voice" indicator.

Truth and memory are two constellations; switching between them, or a compare mode, shows the gap.

### 4.2 The time scrubber: "What was true, told and remembered at this moment?" (P1, global)
- One clock drives everything, with a visible day/night cycle.
- Scrubbing = checking out a commit. Every view re-renders to that state.
- **Nights are dreams.** At each simulated midnight, play the consolidation as a short, calm animation: cards shuffle and merge, "next Saturday" ink rewrites itself into a date, past plans fade. Show the diff it made.
- Mark sessions, dreams, fact changes and town events on the timeline track.

### 4.3 Mind view, in first person: "What does the assistant actually see?" (P1)
**Be the assistant at time t.** Render the literal context window:
- `<profile>`, `<preferences>`, `<memory_listing>` (paths, descriptions, aliases);
- the recent sessions;
- a token-budget bar.

Then a probe arrives. Watch the assistant decide which files to open (`memory_read`) and write its answer. Then the checks appear one by one beside a **truth panel** that only this view's compare mode reveals.

Show three answers side by side, **no memory | this system | oracle**, with the memory-gain-captured bar between them.

This view is the clearest explanation of what we test. Make it beautiful.

### 4.4 Town view: "How is the whole village being understood?" (P2)
- A 2.5D isometric or gently stylized map of the fictional town: high school, community college, the state-university campus nearby, hospital, main street, the warehouse.
- Residents move along their life-log schedules. Relationship lines show on hover.
- Town events appear as ambient changes: heat-wave haze, a lit gym on the night of the school play.
- **Understanding halos:** each principal carries a soft light whose fullness is their current Understanding score, and whose clarity is their Integrity. As the season runs, the village lights up as the assistants come to know people. A principal whose memory is corrupting visibly blurs.
- While a session is happening, a small phone-glow at the principal's location shows someone talking to their assistant.
- **"Be the resident" (optional, first person):** a day in one principal's life, told as a diary, with their phone showing the assistant chat. The viewer feels how little of a day gets told.

### 4.5 Evolution view: "What made the memory system better?" (P2)
- **The lineage garden:** every candidate is a node. Accepted candidates form the trunk; rejected ones are faded side branches. Each node shows its one-line diff and a five-dimension delta glyph: a small radar or five ticks.
- Click a node to see the diff, the per-dimension deltas with confidence intervals, the probes that flipped, and the integrity lines that changed.
- **"Causes" panel:** group candidates by the setting they changed (consolidation cadence, consolidation style, profile cap, recent window, inference policy, and so on) and show the average effect per dimension. This is the panel that answers the project's question.
- **Two minds:** pick two versions and the same principal at the same moment, and diff their memories side by side. Mark the first commit where they diverged.

### 4.6 Forensics: "Where did this wrong memory come from?" (P3)
**Blame for memory.** Click any wrong line to see:
- the commit that introduced it;
- the operation that did it (fast write or dream merge);
- the session excerpt it came from;
- the truth it contradicts.

Then walk backward through its edit history.

### 4.7 Lag curve: "How does memory hold up as facts age?" (P3)
Score against lag since disclosure, per dimension and per system version. Mark the clock boundaries: same session, after a dream, outside the recent window.

## 5. Guided tour (P1, the explainer for newcomers)
A narrated walkthrough of one real case from the data, for example:
1. Dana downplays her achievements, shown across 4 episodes and never stated.
2. Watch the `[inferred]` line form after the third episode, with its evidence threads.
3. Watch a dream merge it into the right file.
4. Then the probe "help me write a LinkedIn post about my new role": the no-memory answer is boastful, the system's is understated, and the checks pass.
5. Switch to a rejected candidate where full-rewrite dreaming dropped the line, and watch the same probe fail.

Each step lives on the time scrubber. The tour should be pausable and scrubbable.

## 6. Data contract
The visualization reads a static **replay bundle** (JSON) exported by the backend. It never calls models.

| Part | Status | Contents |
|---|---|---|
| `world.json`, `cast.json` | **pilot exists soon** (`data/villages/pilot-v1/`) | Town, places, town-event calendar; residents, tiers, links |
| `principals/<id>/bible.json` (facts, impressions, references, arcs, voice) | **pilot exists soon** | Truth layers |
| `principals/<id>/life_log.jsonl`, `disclosure_plan.jsonl`, `sessions/*.json` | **pilot exists soon** | Lived and Told layers |
| `systems/<version>/commits.jsonl` + memory file contents per commit | TBD (system v0 track) | Remembered layer; per-commit operation, time and session id; integrity labels per line |
| `probes.json`, `systems/<version>/probe_results.jsonl` | TBD | Request, t, targets, the three answers, files read, per-check verdicts |
| `systems/<version>/scorecard.json`, `candidates.jsonl` | TBD | Five dimensions with CIs; diff, deltas, accepted or rejected, parent |

Schemas are defined in code under `village/schema.py` (generator). **Build against the pilot files now** (Lived and Told), stub the TBD parts with a small, hand-made fixture that matches the descriptions above, and report the fields you needed so the backend can export them.

## 7. Visual language
- **Mood:** a calm, editorial village at dusk. Warm truth, cool translucent memory. Quiet motion. Nothing flashes unless something went wrong.
- **Typography:** legible, editorial. Transcripts read like real chats; memory files read like real markdown, with frontmatter visible on demand.
- **Motion has meaning:** ink drawing in means a line was filed; a shuffle means a dream; fading means staleness; a blur means corruption. No decorative motion.
- **Accessibility:** status is never color-only; respect reduced motion; everything is keyboard-scrubbable.

## 8. Technical suggestions (not requirements)
- Web: React + TypeScript; one global time store (e.g. zustand) that drives every view.
- Town: react-three-fiber or PixiJS for the isometric map. Charts and threads: D3 or visx. Transcripts: virtualized lists.
- Scrubbing must feel instant. Precompute per-commit diffs and per-day aggregates in the bundle, and load principals lazily.
- Deterministic: the same bundle and the same time always render the same frame.

## 9. Rules (must and must not)
- **MUST keep the epistemic separation strict.** The assistant's view (mind view, Remembered layer) never shows truth unless the user enters an explicit compare mode, and that mode is visibly marked.
- **MUST make every number traceable** to its source file, commit or probe in one click.
- **MUST answer each view's named question;** cut any element that doesn't.
- **MUST NOT invent data.** Missing bundle parts show as "not yet generated", never as plausible fakes, except clearly labeled dev fixtures.
- **MUST NOT show never-store content.** It renders as a redaction bar everywhere, including in truth views. A minor principal's own age and date of birth are never shown.

## 10. The bar: done when a newcomer, after the 5-minute guided tour, can explain
1. what the assistant can and cannot know about a person, and why most of life is never told;
2. how one specific memory line came to exist (its evidence) and how a dream changed it;
3. what a probe tests, and why one passed and another failed;
4. what changed between two system versions, and what that did to each scorecard dimension.

Test the tour on someone who hasn't read the spec. If they can't explain all four, the design isn't done.

## 11. Open questions to bring back
- Is 2.5D isometric or a stylized flat map the better base for the town?
- Can the person constellation and the three-band timeline be one view, or must they be two?
- Which fields are missing from the bundle to draw each view? List them.
