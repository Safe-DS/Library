import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


def test_create_timeseries() -> None:
    ts = TimeSeries(data={"f1": [1, 2, 3, 4, 6], "target": [7,2, 3, 1, 3], "f2": [4,7, 5, 5, 5]},
                    target_name="target",
                    date_name="f1",
                    window_size=1,
                    feature_names=["f1", "f2"])
