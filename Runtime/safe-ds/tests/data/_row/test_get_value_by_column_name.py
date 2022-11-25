import pandas as pd
import pytest
from safe_ds.data import Table
from safe_ds.exceptions import ColumnNameError


def test_get_value_by_column_name_invalid():
    with pytest.raises(ColumnNameError):
        table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
        row = table.get_row_by_index(0)
        row.get_value_by_column_name("col3")


def test_get_value_by_column_name_valid():
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
    row = table.get_row_by_index(0)
    assert (
        row.get_value_by_column_name("col1") == 1
        and row.get_value_by_column_name("col2") == 2
    )
