import pandas as pd
from safe_ds.data import Table


def test_mode_valid():
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4, 3]}))
    column = table.get_column("col1")
    assert column.statistics.mode() == 3


def test_mode_valid_str():
    table = Table(pd.DataFrame(data={"col1": ["1", "2", "3", "4", "3"]}))
    column = table.get_column("col1")
    assert column.statistics.mode() == "3"
