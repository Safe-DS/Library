import pandas as pd
import pytest
from safeds.data import Column, Table
from safeds.exceptions import UnknownColumnNameError


def test_get_column_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    assert (
        isinstance(table.get_column("col1"), Column)
        and table.get_column("col1")._data[0] == pd.Series(data=["col1_1"])[0]
        and table.get_column("col1")._data[0] == "col1_1"
    )


def test_get_column_invalid() -> None:
    with pytest.raises(UnknownColumnNameError):
        table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
        table.get_column("col3")
