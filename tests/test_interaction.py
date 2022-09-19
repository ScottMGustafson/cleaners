from unittest import mock

import pytest

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


@mock.patch("yaml.full_load")
@mock.patch("cleaners.interaction._get_top_feats")
def test_get_interaction_subset(mock_top_feats, mock_yaml_load):

    obj = interaction.TwoWayInteractions(
        max_interact_feats=3,
        interact_top_n=3,
        seed=0,
        add_new_feats=True,
        subset=[
            0,
        ],
    )

    obj.interact_top_n = 3
    obj.feature_dict_path = "fake/path1"
    mock_top_feats.return_value = [1, 2, 3, 3]
    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        obj._get_interact_subset()
        mock_file.assert_called_with("fake/path1", 'r')
        assert obj.subset == [0, 1, 2, 3]

    obj.subset = [0]
    obj.feature_dict_path = None
    obj.interact_feature_path = "fake/path2"
    mock_yaml_load.return_value = [5, 7, 7, 6]
    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        obj._get_interact_subset()
        mock_file.assert_called_with("fake/path2", 'r')
        assert obj.subset == [0, 5, 6, 7]

    obj.feature_dict_path = None
    obj.interact_feature_path = None

    with pytest.raises(Exception):
        obj._get_interact_subset()


def test_sample_list():
    obj = interaction.TwoWayInteractions(
        max_interact_feats=3, interact_top_n=3, seed=0, add_new_feats=True, subset=[0]
    )
    lst = obj._sample_list(list(range(100)))
    assert len(lst) == 3

    lst = sorted(obj._sample_list(list(range(3))))
    assert lst == [0, 1, 2]
