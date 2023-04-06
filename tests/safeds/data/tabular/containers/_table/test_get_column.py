import pandas as pd
import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


def test_get_column_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    assert isinstance(table.get_column("col1"), Column)
    assert table.get_column("col1")._data[0] == pd.Series(data=["col1_1"])[0]
    assert table.get_column("col1")._data[0] == "col1_1"


def test_get_column_invalid() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    with pytest.raises(UnknownColumnNameError):
        table.get_column("col3")
