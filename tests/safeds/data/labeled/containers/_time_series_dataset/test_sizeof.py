import sys

import pytest
from safeds.data.labeled.containers import TimeSeriesDataset


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TimeSeriesDataset(
            {
                "feature_1": [3, 9, 6],
                "feature_2": [6, 12, 9],
                "target": [1, 3, 2],
                "time": [1, 2, 3],
            },
            "target",
            "time",
            window_size=1,
        ),
        TimeSeriesDataset(
            {
                "feature_1": [3, 9, 6],
                "feature_2": [6, 12, 9],
                "other": [3, 9, 12],
                "target": [1, 3, 2],
                "time": [1, 2, 3],
            },
            "target",
            "time",
            window_size=1,
            extra_names=["other"],
        ),
    ],
    ids=["normal", "table_with_extra_column"],
)
def test_should_size_be_greater_than_normal_object(tabular_dataset: TimeSeriesDataset) -> None:
    assert sys.getsizeof(tabular_dataset) > sys.getsizeof(object())
