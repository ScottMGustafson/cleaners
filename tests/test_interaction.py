from cleaners.interaction import _concat_list_of_tuples, get_expected_pairs


def test_exp_cols():
    feat_pairs = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "d")]
    feats = ["a", "b", "c", "a_X_b", "a_X_d"]

    exp_pairs = get_expected_pairs(feat_pairs, feats)
    assert exp_pairs == [["a", "b"], ["a", "d"]]


def test_concat_list_of_tuples():
    lst1 = [("a", "b"), ("b", "a"), ("c", "e")]
    lst2 = [["a", "b"], ("c", "a"), ("c", "b")]
    allowed_list = ["a", "b", "c", "d"]
    res = _concat_list_of_tuples(lst1, lst2, allowed_list)
    assert res == [["a", "b"], ["a", "c"], ["b", "c"]]
