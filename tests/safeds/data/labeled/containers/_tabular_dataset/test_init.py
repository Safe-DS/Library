import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("data", "target_name", "extra_names", "error", "error_msg"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            ["D", "E"],
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
            [],
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
            ["A"],
            ValueError,
            r"Column 'A' cannot be both target and extra.",
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            ["A", "B", "C"],
            ValueError,
            r"At least one feature column must remain.",
        ),
        (
            {
                "A": [1, 4],
            },
            "A",
            [],
            ValueError,
            r"At least one feature column must remain.",
        ),
    ],
    ids=[
        "extra_does_not_exist",
        "target_does_not_exist",
        "target_and_extra_overlap",
        "features_are_empty_explicitly",
        "features_are_empty_implicitly",
    ],
)
def test_should_raise_error(
    data: dict[str, list[int]],
    target_name: str,
    extra_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TabularDataset(data, target_name=target_name, extra_names=extra_names)


@pytest.mark.parametrize(
    ("data", "target_name", "extra_names"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
            [],
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
    ids=[
        "create_tabular_dataset",
        "tabular_dataset_not_all_columns_are_features",
        "tabular_dataset_with_extra_names_as_None",
    ],
)
def test_should_create_a_tabular_dataset(
    data: dict[str, list[int]],
    target_name: str,
    extra_names: list[str] | None,
) -> None:
    tabular_dataset = TabularDataset(data, target_name=target_name, extra_names=extra_names)
    table = Table(data)

    if extra_names is None:
        extra_names = []

    assert isinstance(tabular_dataset, TabularDataset)
    assert tabular_dataset._extras.column_names == extra_names
    assert tabular_dataset._target.name == target_name
    assert tabular_dataset._extras == table.keep_only_columns(extra_names)
    assert tabular_dataset._target == table.get_column(target_name)
