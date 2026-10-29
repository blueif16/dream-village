# How memory benchmarks simulate user ↔ assistant sessions (2026-10-29)

13 benchmarks, from each paper's method/appendix (arXiv HTML; MSC from the ACL PDF). "Same call" = one prompt writes
both sides. Two claims re-checked by hand against the papers (LongMemEval A.1; AMemGym §4.2, App. C.2).

| Benchmark | User side: writer · sees | Assistant side: writer · sees | World · scale | Verification · rates |
|---|---|---|---|---|
| LoCoMo 2402.17753 | one gpt-3.5-turbo agent per speaker · own persona, own dated events, own summaries | no assistant; a second agent of the same kind | independent pairs; 50 convs, ≤35 sessions | humans edited ~15% of turns |
| MSC 2107.07567 | crowdworker with a persona · all earlier dialogue + summaries | no assistant; a second crowdworker | 1,155 personas; ≤5 sessions | onboarding, per-turn flags, filtering |
| LongMemEval 2410.10813 | Llama 3 70B user LLM · background, one evidence statement to convey indirectly, others to avoid; not the question | same model, **no assistant prompt, the dialogue only** | no persistent user; stitched histories | **~70% of sessions human-edited**; user LLM sometimes took the assistant role |
| PerLTQA 2402.16288 | same call · profile, event details | same call | 141 characters, 3,409 dialogues | sample-and-refine, LLM QA scoring |
| MemSim 2409.20163 | LLM rewrites a structured hint into one user message | none (user messages only) | profiles sampled by a Bayesian net; 26k messages | humans on ~20%: 99.5–99.8% correct; plain-LLM gold <90% |
| PrefEval 2502.09597 | same call (Claude 3.5 Sonnet) · preference + persona | same call | 1,000 preferences, 100 personas | ~10k generated → 1k kept |
| PersonaMem 2504.14225 | same call (GPT-4o) · persona, dated history, preference updates | same call | 20 personas, 180+ histories | self-reflection for missed events; human study 90–98% |
| MemBench 2506.21605 | same call · user entity info | same call; prompt says the assistant "does not know the information" | 500 user graphs | none reported |
| **AMemGym 2603.01966** | gpt-4.1 user simulator · profile, **pre-generated utterances carrying state**, recent turns, schema marked DO NOT reveal | **the system under test, on-policy** · the dialogue only | 20 profiles × 10 questions | state exposure 99.1%, integrity 99.2%; gold–human 0.96 vs human–human 0.92 |
| BEAM 2510.27246 | LLaMA-3.3-70B, user questions in batches · plan bullets | separate role-play LLM · plan + summaries | 100 convs, 128K–10M tokens | human Likert 4.5–4.6 |
| HaluMem 2511.03506 | unverified | unverified | 20 users | 95.7% of 700 annotated sessions correct |
| DynamicMem 2606.22877 | same call (gemini-3-flash) · profile, intent, app state | same call | 10 personas; world generated per user | 97–98% coherence; LLM reviser + human escalation |
| PersonaBench 2502.20616 | GPT-4o, user–user chats | unverified | **shared** social communities | none |

## Findings
- One-call scripting is common (PersonaMem, PrefEval, MemBench, PerLTQA, DynamicMem) and is patched with instructions
  and self-reflection ("the assistant does not know the information"; missed events); none measures assistant-side
  leakage. The ones that keep the assistant blind: AMemGym (the system under test) and LongMemEval (an unprompted LLM
  seeing only the dialogue). Two agents aren't free of problems: LongMemEval still human-edited ~70% of sessions.
- A per-person generator thread (profile → dated events → turns) is the norm: LoCoMo, PersonaMem, HaluMem,
  DynamicMem, AMemGym, BEAM.
- Users are almost always independent (10–500); only PersonaBench shares a world, and reports no consistency numbers.
- **AMemGym §4.2: "Off-policy evaluation introduces reuse bias, undermining memory optimization and configuration
  selection"**: replaying fixed traces changed the rankings of memory implementations (plain-LLM rankings mostly held).
- Cost (AMemGym App. C.2): the simulated user is cheap ($0.17 per instance with gpt-4.1, $0.02 with deepseek-v3);
  the assistant under test is the cost (~$13 for gpt-4.1).
