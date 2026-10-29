import numpy as np

from memobench.calibrate import choose_threshold, max_prior_similarity


def test_first_candidate_has_no_prior_and_never_passes():
    s = max_prior_similarity(np.eye(3), batches=[0, 1, 2])
    assert s[0] == -1.0


def test_orthogonal_candidates_score_zero_and_duplicates_score_one():
    v = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    s = max_prior_similarity(v, batches=[0, 1, 2])
    assert s[1] == 0.0 and s[2] == 1.0


def test_similarity_is_cosine_not_dot_product():
    v = np.array([[0.5, 0], [0.5, 0]], dtype=float)  # dot = 0.25, cosine = 1
    assert max_prior_similarity(v, batches=[0, 1])[1] == 1.0


def test_same_batch_candidates_do_not_count_as_prior():
    # Memora only merges with entries from other extraction batches (memory_builder.py:336-363)
    v = np.array([[1, 0], [1, 0]], dtype=float)
    assert max_prior_similarity(v, batches=[0, 0])[1] == -1.0


def test_threshold_hits_target_pass_rate():
    s = np.linspace(0, 1, 101)  # 101 distinct scores
    t, rate = choose_threshold(s, target=0.23)
    assert 0.21 <= rate <= 0.25
    assert rate == np.mean(s >= t)
