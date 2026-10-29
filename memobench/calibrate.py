"""Consolidation-threshold calibration (addendum §5): pick t so ~21-25% of candidates would reach an earlier one."""
import numpy as np


def max_prior_similarity(vectors, batches) -> np.ndarray:
    """s_i = max cosine(candidate i, any earlier candidate from a different batch); -1 when there is none."""
    v = np.asarray(vectors, dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    batches = np.asarray(batches)
    sims = v @ v.T
    s = np.full(len(v), -1.0)
    for i in range(1, len(v)):
        prior = np.flatnonzero(batches[:i] != batches[i])
        if prior.size:
            s[i] = float(np.clip(sims[i, prior].max(), -1.0, 1.0))
    return s


def choose_threshold(s, target: float = 0.23) -> tuple[float, float]:
    """Return (t, achieved pass rate) where pass = s >= t, choosing the candidate score closest to target."""
    s = np.asarray(s, dtype=float)
    rates = {float(t): float(np.mean(s >= t)) for t in np.unique(s)}
    t = min(rates, key=lambda t: abs(rates[t] - target))
    return t, rates[t]
