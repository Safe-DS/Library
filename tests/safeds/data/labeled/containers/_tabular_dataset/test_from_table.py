import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "target_name", "extra_names", "error", "error_msg"),
    [
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            ["D", "E"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D, E'",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "D",
            [],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D'",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "A",
            ["A"],
            ValueError,
            r"Column 'A' cannot be both target and extra.",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            ["A", "B", "C"],
            ValueError,
            r"At least one feature column must remain.",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                },
            ),
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
    table: Table,
    target_name: str,
    extra_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TabularDataset._from_table(table, target_name=target_name, extra_names=extra_names)


@pytest.mark.parametrize(
    ("table", "target_name", "extra_names"),
    [
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            [],
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            ["A", "C"],
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
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
def test_should_create_a_tabular_dataset(table: Table, target_name: str, extra_names: list[str] | None) -> None:
    tabular_dataset = TabularDataset._from_table(table, target_name=target_name, extra_names=extra_names)

    if extra_names is None:
        extra_names = []

    assert isinstance(tabular_dataset, TabularDataset)
    assert tabular_dataset._extras.column_names == extra_names
    assert tabular_dataset._target.name == target_name
    assert tabular_dataset._extras == table.keep_only_columns(extra_names)
    assert tabular_dataset._target == table.get_column(target_name)
