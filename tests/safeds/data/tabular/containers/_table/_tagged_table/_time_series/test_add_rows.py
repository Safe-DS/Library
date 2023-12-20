import pytest
from safeds.data.tabular.containers import Row, Table, TimeSeries
from safeds.exceptions import UnknownColumnNameError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("time_series", "rows", "expected"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1],
                    "feature": [0, 1],
                    "target": [4, 5],
                },
                "target",
                "time",
            ),
            [
                Row(
                    {
                        "time": 2,
                        "feature": 2,
                        "target": 6,
                    },
                ),
                Row({"time": 3, "feature": 3, "target": 7}),
            ],
            TimeSeries(
                {
                    "time": [0, 1, 2, 3],
                    "feature": [0, 1, 2, 3],
                    "target": [4, 5, 6, 7],
                },
                "target",
                "time",
            ),
        ),
    ],
    ids=["add_rows"],
)
def test_should_add_rows(time_series: TimeSeries, rows: list[Row], expected: TimeSeries) -> None:
    assert_that_time_series_are_equal(time_series.add_rows(rows), expected)


@pytest.mark.parametrize(
    ("time_series", "rows", "error_msg"),
    [
        (
            TimeSeries({"time": [], "feature": [], "target": []}, "target", "time", ["feature"]),
            [Row({"feat": None, "targ": None}), Row({"targ": None, "feat": None})],
            r"Could not find column\(s\) 'time, feature, target'.",
        ),
    ],
    ids=["columns_missing"],
)
def test_should_raise_an_error_if_rows_schemas_are_invalid(
    time_series: TimeSeries,
    rows: list[Row] | Table,
    error_msg: str,
) -> None:
    with pytest.raises(UnknownColumnNameError, match=error_msg):
        time_series.add_rows(rows)
