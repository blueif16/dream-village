"""Addendum §5: calibrate Memora's consolidation threshold t for gemini-embedding-2 on one LoCoMo conversation.

Runs the normal build path with consolidation disabled: every extracted candidate is recorded (and stored
as a new entry) instead of merged. Segmentation / extraction / cue calls hit the LLM response cache on the
real build later, so this costs nothing twice. Then embeds candidates with the consolidation prefix and
picks t so ~23% of candidates reach an earlier candidate from another batch (Memora's merge rule).

Usage (from vendor/Memora/app/locomo):
  MEMOBENCH_RUN_ID=calib-c1 uv run python ../../../../scripts/calibrate_threshold.py \
      memory.memory_store=calib-c1 'eval.conversations=[1]' memory.force_rebuild=true
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCOMO = REPO / "vendor/Memora/app/locomo"
for p in (REPO, REPO / "vendor/Memora/src", LOCOMO):
    sys.path.insert(0, str(p))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from memobench.calibrate import choose_threshold, max_prior_similarity  # noqa: E402
from memobench.embed import embed  # noqa: E402
from memora.builder.memory_builder import MemoryBuilder  # noqa: E402
from providers.memora.add import MemoraADD  # noqa: E402

CANDIDATES: list[tuple[str, str, str]] = []  # (user_id, creation_time batch, index text)
_orig_add = None


def _record_and_add(self, entry, *a, **k):
    try:
        self.agent_memory.add(entry)  # store as new, never merge (no update LLM calls)
    except AssertionError:
        return  # exact duplicate index: upstream skips it before the threshold check (memory_builder.py:387-408)
    CANDIDATES.append((self.agent_memory.get_user_id(), str(entry.creation_time), entry.index))


@hydra.main(version_base=None, config_path=str(LOCOMO / "conf"), config_name="config")
def main(cfg: DictConfig):
    MemoryBuilder.upsert_memory_entry = _record_and_add
    MemoraADD(cfg, data_path=str(LOCOMO / "data/locomo10.json")).process_all_conversations()

    out = {"n_candidates": len(CANDIDATES), "per_user": {}}
    all_s = []
    for user in sorted({u for u, _, _ in CANDIDATES}):
        rows = [(b, t) for u, b, t in CANDIDATES if u == user]  # session order preserved
        vecs = embed([t for _, t in rows], kind="consolidation")
        s = max_prior_similarity(np.array(vecs), batches=[b for b, _ in rows])
        all_s.append(s)
        out["per_user"][user] = len(rows)
    s = np.concatenate(all_s)
    t, rate = choose_threshold(s, target=0.23)
    hist, edges = np.histogram(s[s > -1], bins=20, range=(0, 1))
    out.update(t=t, pass_rate=rate, rate_at_0_8=float(np.mean(s >= 0.8)),
               histogram={"edges": edges.round(2).tolist(), "counts": hist.tolist()})
    dest = REPO / "runs/calibration" / f"{cfg.memory.memory_store}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_candidates", "per_user", "t", "pass_rate", "rate_at_0_8")}, indent=2))
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
