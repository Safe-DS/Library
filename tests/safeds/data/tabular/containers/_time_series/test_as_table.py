import pytest
from safeds.data.tabular.containers import Table, TimeSeries


@pytest.mark.parametrize(
    ("time_series", "expected"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
                "time",
                ["feature_1", "feature_2"],
            ),
            Table(
                {
                    "time": [0, 1, 2],
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
                "target",
                "time",
                ["feature_1", "feature_2"],
            ),
            Table(
                {
                    "time": [0, 1, 2],
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
            ),
        ),
    ],
    ids=["normal", "table_with_column_as_non_feature"],
)
def test_should_return_table(time_series: TimeSeries, expected: Table) -> None:
    table = time_series._as_table()
    assert table.schema == expected.schema
    assert table == expected
