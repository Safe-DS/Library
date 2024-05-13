import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


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
                ColumnNotFoundError,
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
                ColumnNotFoundError,
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
                ColumnNotFoundError,
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
                ColumnNotFoundError,
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
        "dict_extra_does_not_exist",
        "dict_target_does_not_exist",
        "dict_target_and_extra_overlap",
        "dict_features_are_empty_explicitly",
        "dict_features_are_empty_implicitly",
        "table_extra_does_not_exist",
        "table_target_does_not_exist",
        "table_target_and_extra_overlap",
        "table_features_are_empty_explicitly",
        "table_features_are_empty_implicitly",
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
        "dict_create_tabular_dataset",
        "dict_tabular_dataset_not_all_columns_are_features",
        "dict_tabular_dataset_with_extra_names_as_None",
        "table_create_tabular_dataset",
        "table_tabular_dataset_not_all_columns_are_features",
        "table_tabular_dataset_with_extra_names_as_None",
    ],
)
def test_should_create_a_tabular_dataset(
    data: Table | dict[str, list[int]],
    target_name: str,
    extra_names: list[str] | None,
) -> None:
    tabular_dataset = TabularDataset(data, target_name=target_name, extra_names=extra_names)
    if not isinstance(data, Table):
        data = Table(data)

    if extra_names is None:
        extra_names = []

    assert isinstance(tabular_dataset, TabularDataset)
    assert tabular_dataset._extras.column_names == extra_names
    assert tabular_dataset._target.name == target_name
    assert tabular_dataset._extras == data.remove_columns_except(extra_names)
    assert tabular_dataset._target == data.get_column(target_name)
