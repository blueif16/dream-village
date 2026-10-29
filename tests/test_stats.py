from memobench.stats import cluster_bootstrap, paired_cluster_bootstrap


def test_point_estimate_is_micro_accuracy_over_all_items():
    # 3 of 4 correct in conv a, 1 of 2 in conv b -> 4/6 pooled, not the mean of per-conv rates (0.625)
    est, lo, hi = cluster_bootstrap({"a": [1, 1, 1, 0], "b": [1, 0]}, n_boot=200, seed=0)
    assert est == 4 / 6
    assert lo <= est <= hi


def test_constant_scores_give_degenerate_interval():
    est, lo, hi = cluster_bootstrap({"a": [1, 1], "b": [1, 1, 1]}, n_boot=200, seed=0)
    assert (est, lo, hi) == (1.0, 1.0, 1.0)


def test_clustered_interval_is_wider_than_iid_when_clusters_are_homogeneous():
    # 10 conversations, each all-correct or all-wrong: items are not independent
    clustered = {f"c{i}": [i % 2] * 20 for i in range(10)}
    _, lo_c, hi_c = cluster_bootstrap(clustered, n_boot=2000, seed=0)
    iid = {f"q{j}": [s] for j, s in enumerate(s for v in clustered.values() for s in v)}
    _, lo_i, hi_i = cluster_bootstrap(iid, n_boot=2000, seed=0)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_same_seed_same_interval():
    data = {"a": [1, 0, 1], "b": [0, 0, 1], "c": [1, 1, 1]}
    assert cluster_bootstrap(data, n_boot=500, seed=7) == cluster_bootstrap(data, n_boot=500, seed=7)


def test_paired_diff_of_identical_systems_is_exactly_zero():
    a = {"a": [1, 0, 1], "b": [0, 1]}
    assert paired_cluster_bootstrap(a, a, n_boot=300, seed=0) == (0.0, 0.0, 0.0)


def test_paired_diff_point_estimate_is_b_minus_a_on_same_items():
    a = {"a": [0, 0, 1, 1], "b": [0, 1]}
    b = {"a": [1, 0, 1, 1], "b": [1, 1]}  # two extra correct out of 6
    est, lo, hi = paired_cluster_bootstrap(a, b, n_boot=300, seed=0)
    assert est == 2 / 6
    assert lo <= est <= hi and lo >= 0


def test_paired_requires_same_items():
    import pytest

    with pytest.raises(ValueError):
        paired_cluster_bootstrap({"a": [1, 0]}, {"a": [1]}, n_boot=10, seed=0)
