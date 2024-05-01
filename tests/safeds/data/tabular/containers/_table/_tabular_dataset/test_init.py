import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("data", "target_name", "feature_names", "error", "error_msg"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            ["A", "B", "C", "D", "E"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D, E'",
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "D",
            ["A", "B", "C"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D'",
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "A",
            ["A", "B", "C"],
            ValueError,
            r"Column 'A' cannot be both feature and target.",
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "D",
            [],
            ValueError,
            r"At least one feature column must be specified.",
        ),
        (
            {
                "A": [1, 4],
            },
            "A",
            None,
            ValueError,
            r"At least one feature column must be specified.",
        ),
    ],
    ids=[
        "feature_does_not_exist",
        "target_does_not_exist",
        "target_and_feature_overlap",
        "features_are_empty-explicitly",
        "features_are_empty_implicitly",
    ],
)
def test_should_raise_error(
    data: dict[str, list[int]],
    target_name: str,
    feature_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TabularDataset(data, target_name=target_name, feature_names=feature_names)


@pytest.mark.parametrize(
    ("data", "target_name", "feature_names"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            ["A", "B", "C"],
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            ["A", "C"],
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            None,
        ),
    ],
    ids=["create_tabular_dataset", "tabular_dataset_not_all_columns_are_features", "tabular_dataset_with_feature_names_as_None"],
)
def test_should_create_a_tabular_dataset(
    data: dict[str, list[int]],
    target_name: str,
    feature_names: list[str] | None,
) -> None:
    tabular_dataset = TabularDataset(data, target_name=target_name, feature_names=feature_names)
    if feature_names is None:
        feature_names = list(data.keys())
        feature_names.remove(target_name)
    assert isinstance(tabular_dataset, TabularDataset)
    assert tabular_dataset._features.column_names == feature_names
    assert tabular_dataset._target.name == target_name
    assert tabular_dataset._features == Table(data).keep_only_columns(feature_names)
    assert tabular_dataset._target == Table(data).get_column(target_name)
