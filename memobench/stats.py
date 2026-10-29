"""Cluster bootstrap CIs: resample clusters (conversations / users), score = pooled micro-average over items."""
import numpy as np


def _arrays(scores: dict[str, list[float]]):
    keys = sorted(scores)
    return keys, [np.asarray(scores[k], dtype=float) for k in keys]


def _ci(stat, n_clusters, n_boot, seed, alpha):
    rng = np.random.default_rng(seed)
    boots = [stat(rng.integers(0, n_clusters, n_clusters)) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def cluster_bootstrap(scores, *, n_boot=10_000, seed=0, alpha=0.05):
    """scores: {cluster_id: [item scores]} -> (estimate, lo, hi)."""
    _, arrs = _arrays(scores)
    sums = np.array([a.sum() for a in arrs])
    ns = np.array([len(a) for a in arrs])
    est = float(sums.sum() / ns.sum())
    lo, hi = _ci(lambda idx: sums[idx].sum() / ns[idx].sum(), len(arrs), n_boot, seed, alpha)
    return est, lo, hi


def paired_cluster_bootstrap(a, b, *, n_boot=10_000, seed=0, alpha=0.05):
    """Paired difference b - a on the same items, resampling clusters -> (estimate, lo, hi)."""
    if sorted(a) != sorted(b) or any(len(a[k]) != len(b[k]) for k in a):
        raise ValueError("paired comparison needs the same clusters and items in both systems")
    return cluster_bootstrap(
        {k: np.asarray(b[k], dtype=float) - np.asarray(a[k], dtype=float) for k in a},
        n_boot=n_boot, seed=seed, alpha=alpha,
    )
