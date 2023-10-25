import pytest
import pandas as pd
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


def test_create_timeseries() -> None:
    table = Table(data={"f1": [1, 2, 3, 4, 6, 7], "target": [7,2, 3, 1, 3, 7], "f2": [4,7, 5, 5, 5, 7]})
    ts = TimeSeries(data={"f1": [1, 2, 3, 4, 6, 7], "target": [7,2, 3, 1, 3, 7], "f2": [4,7, 5, 5, 5, 7]},
                    target_name="target",
                    date_name="f1",
                    window_size=1,
                    feature_names=["f1", "f2"])

    ts_2 = TimeSeries(data={"f1": [1, 2, 3, 4, 5, 6]},
                    target_name="f1",
                    date_name="f1",
                    window_size=2,
                    feature_names=["f1"])
    #assert ts_2._target == Column(name = "f1", data = [[3],[4],[5],[6]] )

    #assert
