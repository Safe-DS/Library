import pandas as pd
import pytest
from safe_ds.data import Column, Table
from safe_ds.exceptions import ColumnNameError


def test_get_column_by_name_valid():
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    assert (
        isinstance(table.get_column_by_name("col1"), Column)
        and table.get_column_by_name("col1")._data[0] == pd.Series(data=["col1_1"])[0]
        and table.get_column_by_name("col1")._data[0] == "col1_1"
    )


def test_get_column_by_name_invalid():
    with pytest.raises(ColumnNameError):
        table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
        table.get_column_by_name("col3")
