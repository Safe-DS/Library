import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("data", "target_name", "time_name", "extra_names", "error", "error_msg"),
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
            "time",
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
                "time": [9, 9],
            },
            "D",
            "time",
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
                "time": [9, 9],
            },
            "A",
            "time",
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
                "time": [9, 9],
            },
            "T",
            "time",
            ["A", "time", "C"],
            ValueError,
            r"Column 'time' cannot be both time and extra.",
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
            "time",
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
                    "time": [9, 9],
                },
            ),
            "D",
            "time",
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
                    "time": [9, 9],
                },
            ),
            "A",
            "time",
            ["A"],
            ValueError,
            r"Column 'A' cannot be both target and extra.",
        ),
    ],
    ids=[
        "dict_extra_does_not_exist",
        "dict_target_does_not_exist",
        "dict_target_and_extra_overlap",
        "dict_features_are_empty_explicitly",
        "table_extra_does_not_exist",
        "table_target_does_not_exist",
        "table_target_and_extra_overlap",
    ],
)
def test_should_raise_error(
    data: dict[str, list[int]],
    target_name: str,
    time_name: str,
    extra_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TimeSeriesDataset(data, target_name=target_name, time_name=time_name, extra_names=extra_names)


@pytest.mark.parametrize(
    ("data", "target_name", "time_name", "extra_names"),
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
            "time",
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
            "time",
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
            "time",
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
            "time",
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
            "time",
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
            "time",
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
    time_name: str,
    extra_names: list[str] | None,
) -> None:
    tabular_dataset = TimeSeriesDataset(data, target_name=target_name, time_name=time_name, extra_names=extra_names)
    if not isinstance(data, Table):
        data = Table(data)

    if extra_names is None:
        extra_names = []

    assert isinstance(tabular_dataset, TimeSeriesDataset)
    assert tabular_dataset._extras.column_names == extra_names
    assert tabular_dataset._target.name == target_name
    assert tabular_dataset._extras == data.keep_only_columns(extra_names)
    assert tabular_dataset._target == data.get_column(target_name)
