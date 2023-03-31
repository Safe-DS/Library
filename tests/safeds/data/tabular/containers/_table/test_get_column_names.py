import pandas as pd
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Schema


def test_get_column_names() -> None:
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [1]}))
    assert table.get_column_names() == ["col1", "col2"]


def test_get_column_names_empty() -> None:
    table = Table(pd.DataFrame(), Schema({}))
    assert not table.get_column_names()
