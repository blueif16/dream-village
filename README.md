# Dream Village

A simulated town for measuring and improving an AI assistant's long-term personal memory.

## Premise
The assistants with personal memory (ChatGPT, Claude apps, Claude Code, Codex, Gemini) have converged on one design:
- a short LLM-written profile, always in context;
- an index of markdown memory files, opened on demand;
- writes after each conversation;
- periodic consolidation ("dreaming");
- the raw history kept as evidence;
- rules for time and restraint.

We take that design as the basis and ask how to **optimize on top of it**: the settings, details and techniques that make it a robust memory layer. See `docs/dream-village/design.md` §0.

## How it works
1. **Generate a village with known truth** (`village/`). A town, its residents, and for each principal:
   - a character bible: dated facts, impressions that are shown but never stated, shared references, life arcs;
   - a daily life log;
   - assistant conversations rendered from a disclosure plan, then verified against that plan.
2. **Replay** the conversations through the memory system (`memsys/`). It writes markdown memory after each session and consolidates nightly, and every memory state is a git commit.
3. **Probe** read-only forks of memory at any time t, and **score** the result (`scorer/`). Code computes gold from the truth. Atomic yes/no checks are judged by Claude models, and an integrity audit labels every memory line at every commit. The scorecard has CIs, a noise floor from an identical second run, and baselines: no memory, recent window, full context, oracle.
4. **Optimize:** one-change candidates are screened, then confirmed against the noise floor.

The rest of the repo:
- `viz/`: a read-only inspector with a git-style timeline, the dialogues and memory diffs, plus a pixel-art town view.
- `labels/`: a page for labeling context cards, used to validate the judges.

## Dataset
`data/villages/full13-v1/` holds 12 principals over 13 simulated weeks (2025-08-25 → 2025-11-23): 598 assistant sessions with their world, cast, bibles, life logs, plans and verification results. Everything is synthetic, including the "never-store" items such as account numbers, which are fake and planted to test restraint. `READ_ME_FIRST.md` there describes it, and `manifest.json` has its quality gate and generation cost.

## Quick start
```
uv sync                                                    # or: python -m venv .venv && pip install -e .
python -m memsys.replay --sessions data/villages/full13-v1/sessions --out runs/replays/v0 --config memsys/config/v0.json
python -m scorer.run --village data/villages/full13-v1 --replay runs/replays/v0 --out runs/scorer/v0
python viz/export.py && (cd viz && npm install && npm run dev)   # http://localhost:5173
```
Model calls need `ANTHROPIC_API_KEY` in `.env` (never committed). Decisions and their evidence are in `docs/dream-village/decisions.md` and `evidence.md`.

`vendor/Memora` is Microsoft's MIT-licensed Memora, used in an earlier reproduction (Track A, `docs/results.md`).
