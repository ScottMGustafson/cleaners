from cleaners.interaction import get_expected_pairs


def test_exp_cols():
    feat_pairs = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "d")]
    feats = ["a", "b", "c", "a_X_b", "a_X_d"]

    exp_pairs = get_expected_pairs(feat_pairs, feats)
    assert exp_pairs == [["a", "b"], ["a", "d"]]
