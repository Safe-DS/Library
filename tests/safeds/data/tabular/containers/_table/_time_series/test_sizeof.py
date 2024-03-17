import sys

import pytest
from safeds.data.tabular.containers import TimeSeries


@pytest.mark.parametrize(
    "time_series",
    [
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
    ],
    ids=["normal", "table_with_column_as_non_feature"],
)
def test_should_size_be_greater_than_normal_object(time_series: TimeSeries) -> None:
    assert sys.getsizeof(time_series) > sys.getsizeof(object())
