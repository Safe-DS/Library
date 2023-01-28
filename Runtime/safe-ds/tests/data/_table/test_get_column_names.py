import pandas as pd
from safeds.data import Table, TableSchema


def test_get_column_names() -> None:
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [1]}))
    assert table.get_column_names() == ["col1", "col2"]


def test_get_column_names_empty() -> None:
    table = Table(pd.DataFrame(), TableSchema({}))
    assert not table.get_column_names()
