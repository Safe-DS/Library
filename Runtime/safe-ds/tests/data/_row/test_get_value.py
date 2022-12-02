import pandas as pd
import pytest
from safe_ds.data import Table
from safe_ds.exceptions import ColumnNameError


def test_get_value_invalid():
    with pytest.raises(ColumnNameError):
        table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
        row = table.get_row(0)
        row.get_value("col3")


def test_get_value_valid():
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
    row = table.get_row(0)
    assert row.get_value("col1") == 1 and row.get_value("col2") == 2
