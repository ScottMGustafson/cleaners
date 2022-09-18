from unittest import mock

from cleaners import interaction
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


def test_get_top_feats():
    lst = interaction._get_top_feats(
        5, feat_impt={k: i for i, k in enumerate("abcdefghijklmnop"[::-1])}
    )
    assert lst == ["a", "b", "c", "d", "e"]

    lst = interaction._get_top_feats(5, feat_impt={k: i for i, k in enumerate("abcd"[::-1])})
    assert lst == ["a", "b", "c", "d"]


def test_interaction(make_pd_data):
    df = make_pd_data

    df = df[[x for x in df.columns if "var" in x or "bin" in x]]

    obj = interaction.TwoWayInteractions(
        max_interact_feats=4,
        interact_top_n=3,
        seed=0,
        add_new_feats=True,
        subset=["var_1", "var_2", "var_3"],
    )

    with mock.patch.object(obj, "_get_interact_subset"):
        obj = obj.fit(df)

    _df = obj.transform(df)

    assert [x for x in _df.columns if "_X_" in x] == [
        'var_1_X_var_2',
        'var_1_X_var_3',
        'var_2_X_var_3',
    ]
