import pytest
from safeds.data.tabular.containers import Row, TimeSeries
from safeds.exceptions import UnknownColumnNameError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("time_series", "row", "expected"),
    [
        (
            TimeSeries(
                {
                    "time": [0,1],
                    "feature": [0, 1],
                    "target": [3, 4],
                },
                "target",
                "time",
            ),
            Row(
                {
                    "time": 2,
                    "feature": 2,
                    "target": 5,
                },
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
            ),
        ),
    ],
    ids=["add_row"],
)
def test_should_add_row(time_series: TimeSeries, row: Row, expected: TimeSeries) -> None:
    assert_that_time_series_are_equal(time_series.add_row(row), expected)

@pytest.mark.parametrize(
    ("time_series", "row", "error_msg"),
    [
        (
            TimeSeries({"time": [], "feature": [], "target": []}, "target", "time", ["feature"]),
            Row({"feat": None, "targ": None}),
            r"Could not find column\(s\) 'time, feature, target'.",
        ),
    ],
    ids=["columns_missing"],
)
def test_should_raise_an_error_if_row_schema_invalid(
    time_series: TimeSeries,
    row: Row,
    error_msg: str,
) -> None:
    with pytest.raises(UnknownColumnNameError, match=error_msg):
        time_series.add_row(row)


# the original tests throw a warning here aswell( test_add_row in tagged_table)
@pytest.mark.parametrize(
    ("time_series", "row", "expected_time_series"),
    [
        (
            TimeSeries({"time": [],"feature": [], "target": []}, "target", "time"),
            Row({"time": 0,"feature": 2, "target": 5}),
            TimeSeries({"time": [0], "feature": [2], "target": [5]}, "target", "time"),
        ),
    ],
    ids=["empty_feature_column"],
)
def test_should_add_row_to_empty_table(
    time_series: TimeSeries,
    row: Row,
    expected_time_series: TimeSeries,
) -> None:
    assert_that_time_series_are_equal(time_series.add_row(row), expected_time_series)




