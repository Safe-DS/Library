import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
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
                "time": [9, 9],
            },
            "T",
            ["D", "E"],
            ColumnNotFoundError,
            None,
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
                "time": [9, 9],
            },
            "D",
            [],
            ColumnNotFoundError,
            None,
        ),
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
                "time": [9, 9],
            },
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
                    "time": [9, 9],
                },
            ),
            "T",
            ["D", "E"],
            ColumnNotFoundError,
            None,
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                    "time": [9, 9],
                },
            ),
            "D",
            [],
            ColumnNotFoundError,
            None,
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                    "time": [9, 9],
                },
            ),
            "A",
            ["A"],
            ValueError,
            r"Column 'A' cannot be both target and extra.",
        ),
    ],
    ids=[
        "dict_extra_does_not_exist",
        "dict_target_does_not_exist",
        "dict_target_and_extra_overlap",
        "table_extra_does_not_exist",
        "table_target_does_not_exist",
        "table_target_and_extra_overlap",
    ],
)
def test_should_raise_error(
    data: dict[str, list[int]],
    target_name: str,
    extra_names: list[str] | None,
    error: type[Exception],
    error_msg: str | None,
) -> None:
    with pytest.raises(error, match=error_msg):
        TimeSeriesDataset(data, target_name=target_name, window_size=1, extra_names=extra_names)


@pytest.mark.parametrize(
    ("data", "target_name", "extra_names"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
                "time": [9, 9],
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
                "time": [9, 9],
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
                "time": [9, 9],
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
                    "time": [9, 9],
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
                    "time": [9, 9],
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
                    "time": [9, 9],
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
    tabular_dataset = TimeSeriesDataset(
        data,
        target_name=target_name,
        window_size=1,
        extra_names=extra_names,
    )
    if not isinstance(data, Table):
        data = Table(data)

    if extra_names is None:
        extra_names = []

    assert isinstance(tabular_dataset, TimeSeriesDataset)
    assert tabular_dataset._extras.column_names == extra_names
    assert tabular_dataset._target.name == target_name
    assert tabular_dataset._extras == data.remove_columns_except(extra_names)
    assert tabular_dataset._target == data.get_column(target_name)
