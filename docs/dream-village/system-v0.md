# System v0: the common core (2026-10-28)

This is the starting point of the memory system the Dream Village evolves. It is invented, not ported (D21):
- **The common core** below is what every researched design shares. It is v0's behavior.
- **The settings** are where those designs disagree. They are the loop's search space, and every v0 value is a starting value.
- **The file organization** follows Claude's apps exactly (D23).

v0 is built as a config file plus a small set of prompt files (fast write, dream, answer), so every candidate change is one diff (D19). Evidence keys point to `evidence.md`.

## 1. Common core: what every design shares (D22)
| Part | v0 behavior | Seen in |
|---|---|---|
| Profile, always loaded | A short, LLM-written summary of who the user is, injected every turn, with a hard size cap | ChatGPT's synthesized profile, Claude's `<profile>`, Gemini's summary, Codex's memory summary (E-PROD-1, -5, -7) |
| Index, always loaded | One line per memory file (path, description, aliases) | Claude's `<memory_listing>`, Claude Code's MEMORY.md, Letta's root files (E-PROD-3, -7) |
| Memory files, read on demand | Markdown files the LLM opens when a request needs them | Claude `memory_read`, Claude Code and Grok Build topic files (E-PROD-7) |
| Recent window | The last few sessions, raw | ChatGPT (~40 recent chats), Gemini (recent turns), Claude (recent-chats tool) |
| Raw archive | Every past session kept; searched by keyword as a last resort | All products keep chat history; Claude and ChatGPT search it (E-PROD-2) |
| Fast write | A background pass after each finished turn files what is durable; "remember" and "forget" are applied during the turn | Claude apps, Claude Code, Grok Build (E-PROD-2, -7) |
| Slow consolidation ("dream") | Periodically: merge duplicates, resolve contradictions, turn relative dates into absolute ones, mark plans as past, prune what's unused | ChatGPT Dreaming, Claude Code auto-dream, Codex, ChatGPT dots, Letta (E-PROD-1, -4, -5) |
| Time rules | Lines carry dates; superseded values are marked inline ("tea now (previously coffee)"); a past plan isn't assumed to have happened | ChatGPT month tags, Gemini source dates, Claude supersession, Gemini's rule that plans are not completed events |
| Restraint rules | Don't file a passing mention until it recurs; sensitive categories need consent or are never stored; the user's corrections override | Claude calibration and `<never_store>`, Gemini's corrections ledger (E-PROD-7) |

## 2. Settings: where the designs disagree (D22)
| Setting | The range in the wild | v0 value | Why this start |
|---|---|---|---|
| Consolidation cadence | Every turn (Cognee) · ≥24 h and ≥5 sessions (Claude Code) · nightly (dots) · idle ≥6 h (Codex) · only near the size cap (Claude apps) | Nightly | The middle of the range; matches the day/night clock |
| Consolidation style | Full rewrite · small edits merged by code | Small edits (`str_replace`-style) | Full rewrites are what the evidence shows corrupting memory (E-INT-1, E-INT-2). **Likely first experiment** |
| Profile cap | ~300 words (Claude) · 2,500 tokens (Codex) · ~4k tokens (ChatGPT) | 300 words | Follows Claude (D23) |
| Recent window | None · ~40 chats (ChatGPT) | 10 sessions | A middle value; tune |
| Who picks the memory files | The answering model by tool call (Claude apps) · a small side model, up to 5 files (Claude Code) · everything injected (ChatGPT) | The answering model, by tool call | Follows Claude |
| File organization | By topic, area and person (Claude) · by type (Claude Code) · one profile with sections (ChatGPT) | **Claude apps layout, fixed for v0** (§3) | Shiran (D23). Can be reopened by a new decision |
| Write threshold | File only once it recurs · file anything notable | Only once it recurs, or when the user says "remember" | Follows Claude's calibration rule |
| Stale values | Inline "previously…" · a separate history · deleted | Inline, plus the raw archive as evidence | Follows Claude; keeps the evidence (E-INT-1) |
| Model per step | Any mix | Haiku-tier writes and dreams; a mid-tier model answers | Cheap writes, a capable reader; tune |
| **Inference policy** | Stated facts only (Claude chat: `[stated]` only, never personality inferences) · synthesized understanding (ChatGPT Dreaming's profile) | `[inferred]` allowed for non-sensitive patterns, citing evidence episodes; Claude's never-store kept (D24) | The understanding dimension needs it; "stated-only" is an early experiment |

## 3. File organization (D23)
v0's own spec, modeled on how Claude apps organize memory (E-PROD-7).

**Paths:**
| Path | What goes there |
|---|---|
| `/profile.md` | Who they are, at the level that stays stable: name, role, where they work, what they work on, when they started. The test: would it still be true in three months? Anything dated or "currently" goes in `/areas/` or `/topics/`. **Under 300 words.** Injected every turn as `<profile>` |
| `/preferences.md` | How they want the *assistant* to behave: format, level of detail, what to skip. Injected every turn as `<preferences>`. Not for their tastes; those go in `/topics/` |
| `/topics/<domain>.md` | Facts about them by domain: habits, tastes, routines, time zone, recurring topics, and patterns once they recur (e.g. `food.md`, `schedule.md`, `communication.md`). The fact's domain decides the file, not which files already exist |
| `/areas/<name>.md` | Any ongoing involvement: projects, incidents being handled, recurring responsibilities, chores in progress (an apartment search, a trip). Decisions, constraints, deadlines, current status |
| `/people/<name>.md` | Anyone whose context helps future conversations: their relationship to the user and what they're involved in together. Relationship context, not a dossier. Slug by name or by relationship, with the other handle in `aliases` |

**File format:**
```
---
name: <path stem, unique; what [[links]] resolve against>
description: <one line under 150 characters: what's in the file and the people, places and projects it names; never a sensitive fact>
sources: [chat]
aliases: [other names the subject goes by; durable names only]
---

- [stated] one line per statement, linking other subjects as [[name]]
```

**Rules:**
- **Line tags.** `[stated]` for what the user said; `[observed]` lines are kept when merging. v0 also writes `[inferred]` for non-sensitive patterns (D24).
- **What is excluded:** the model's conclusions, its forward-looking state ("next steps"), its research output, its enrichment of what the user said, hearsay, its own advice, and anything in never-store.
- **Calibration:** a passing mention isn't filed until it recurs. The horizon test: would this line still be true and worth reading a month from now? A task's moving state expires; only its stable residue is filed.
- **One file per subject.** Check the aliases before creating a file, and add new names to an existing file's aliases.
- **Edits:** read before writing. Use `str_replace` for small changes, `write` for new files or restructuring, `append` to add a line. Delete only on an explicit request. Writes carry version tokens.
- **Every turn,** the context gets `<profile>`, `<preferences>` and `<memory_listing>` (path, description, aliases, sources). File bodies come through `memory_read`, several in one call. Always read before saying "I don't have that."
- **Applying memory:** selectively, from none for generic questions to full personalization for personal ones, worked in naturally without narrating the retrieval.
- **Never store,** for anyone, even if asked:
  - government-ID or financial-account numbers;
  - immigration status, caste;
  - **a minor user's own age or date of birth**, which applies to our two teen principals;
  - sexual history, abuse, criminal history;
  - suicide, self-harm, disordered eating;
  - health or personality inferences the user did not state.

  Other protected attributes follow a save-time consent check.

## 4. Departures from Claude (each needs a decision row)
| Departure | Reason | Decision |
|---|---|---|
| Nightly dream (Claude apps has no scheduled consolidation) | Common core across ChatGPT, Claude Code, Codex and Letta; a loop setting | D22 |
| Raw archive searched by keyword | Claude searches past chats with retrieval-augmented generation; we keep v0 free of embeddings | D20 |
| Inference policy: `[inferred]` lines for non-sensitive patterns | The impressions layer (D5) | D24 |
