import pytest
from safeds.data.tabular.containers import TimeSeries


def test_into_dataloader() -> None:
    dataset = TimeSeries(
        {
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
            "feature_1": [3, 9, 6, 3, 9, 6, 3, 9, 6, 3],
            "feature_2": [6, 12, 9, 6, 12, 9, 6, 12, 9, 6],
            "other": [3, 9, 12, 3, 9, 12, 3, 9, 12, 3],
            "target": [1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
        },
        "target", "time", ["feature_1", "feature_2"], )
    dataloader = dataset._into_dataloader_with_window(3, 2, 1)
    assert False


def test_into_dataloader_wo_features() -> None:
    dataset = TimeSeries(
        {
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
            "feature_1": [3, 9, 6, 3, 9, 6, 3, 9, 6, 3],
            "feature_2": [6, 12, 9, 6, 12, 9, 6, 12, 9, 6],
            "other": [3, 9, 12, 3, 9, 12, 3, 9, 12, 3],
            "target": [1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
        },
        "target", "time")
    dataloader = dataset._into_dataloader_with_window(3, 2, 1)
    assert False
