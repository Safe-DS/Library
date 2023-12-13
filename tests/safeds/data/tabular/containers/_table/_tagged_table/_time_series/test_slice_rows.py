import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import IndexOutOfBoundsError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "test_table", "second_test_table"),
    [
        (
            TimeSeries(
                data={"time":[0, 1, 2], "feature": [1, 2, 1], "non_feature": [0, 2, 4], "target": [1, 2, 4]},
                target_name="target",
                time_name="time",
                feature_names=["non_feature"],
            ),
            TimeSeries(
                data={"time":[0, 1],"feature": [1, 2], "non_feature": [0, 2], "target": [1, 2]},
                target_name="target",
                time_name="time",
                feature_names=["non_feature"],
            ),
            TimeSeries(
                {"time":[0, 2], "feature": [1, 1], "non_feature": [0, 4], "target": [1, 4]},
                target_name="target",
                time_name="time",
                feature_names=["non_feature"],
            ),
        ),
    ],
    ids=["Table with three rows"],
)
def test_should_slice_rows(table: TimeSeries, test_table: TimeSeries, second_test_table: TimeSeries) -> None:
    new_table = table.slice_rows(0, 2, 1)
    second_new_table = table.slice_rows(0, 3, 2)
    third_new_table = table.slice_rows()
    assert_that_tagged_tables_are_equal(new_table, test_table)
    assert_that_tagged_tables_are_equal(second_new_table, second_test_table)
    assert_that_tagged_tables_are_equal(third_new_table, table)


@pytest.mark.parametrize(
    ("start", "end", "step", "error_message"),
    [
        (3, 2, 1, r"There is no element in the range \[3, 2\]"),
        (4, 0, 1, r"There is no element in the range \[4, 0\]"),
        (0, 4, 1, r"There is no element at index '4'"),
        (-4, 0, 1, r"There is no element at index '-4'"),
        (0, -4, 1, r"There is no element in the range \[0, -4\]"),
    ],
)
def test_should_raise_if_index_out_of_bounds(start: int, end: int, step: int, error_message: str) -> None:
    table = TimeSeries({"time": [0, 1, 2], "feature": [1, 2, 1], "target": [1, 2, 4]}, "target", "time")

    with raises(IndexOutOfBoundsError, match=error_message):
        table.slice_rows(start, end, step)
