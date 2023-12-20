import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("time_series", "column", "time_series_with_new_column"),
    [
        (
            Table({"t": [1, 2], "f1": [1, 2], "target": [2, 3]}).time_columns(
                target_name="target",
                time_name="t",
                feature_names=["f1"],
            ),
            Column("f2", [4, 5]),
            Table({"t": [1, 2], "f1": [1, 2], "target": [2, 3], "f2": [4, 5]}).time_columns(
                target_name="target",
                time_name="t",
                feature_names=["f1", "f2"],
            ),
        ),
        (
            Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1]}).time_columns(
                target_name="target",
                time_name="other",
                feature_names=["f1"],
            ),
            Column("f2", [4, 5]),
            Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1], "f2": [4, 5]}).time_columns(
                target_name="target",
                time_name="other",
                feature_names=["f1", "f2"],
            ),
        ),
    ],
    ids=["new column as feature", "table contains a non feature/target column"],
)
def test_should_add_column_as_feature(
    time_series: TimeSeries,
    column: Column,
    time_series_with_new_column: TimeSeries,
) -> None:
    assert_that_time_series_are_equal(
        time_series.add_column_as_feature(column),
        time_series_with_new_column,
    )


@pytest.mark.parametrize(
    ("tagged_table", "column", "error_msg"),
    [
        (
            TaggedTable({"A": [1, 2, 3], "B": [4, 5, 6]}, target_name="B", feature_names=["A"]),
            Column("A", [7, 8, 9]),
            r"Column 'A' already exists.",
        ),
    ],
    ids=["column_already_exists"],
)
def test_should_raise_duplicate_column_name_if_column_already_exists(
    tagged_table: TaggedTable,
    column: Column,
    error_msg: str,
) -> None:
    with pytest.raises(DuplicateColumnNameError, match=error_msg):
        tagged_table.add_column_as_feature(column)


# here starts the second test for errors
@pytest.mark.parametrize(
    ("time_series", "column", "error_msg"),
    [
        (
            TimeSeries(
                {"time": [0, 1, 2], "A": [1, 2, 3], "B": [4, 5, 6]},
                target_name="B",
                time_name="time",
                feature_names=["A"],
            ),
            Column("C", [5, 7, 8, 9]),
            r"Expected a column of size 3 but got column of size 4.",
        ),
    ],
    ids=["column_is_oversize"],
)
def test_should_raise_column_size_error_if_column_is_oversize(
    time_series: TimeSeries,
    column: Column,
    error_msg: str,
) -> None:
    with pytest.raises(ColumnSizeError, match=error_msg):
        time_series.add_column_as_feature(column)
