import pandas as pd
from safeds.data.tabular.containers import Row
from safeds.data.tabular.typing import RealNumber, TableSchema


def test_get_column_names() -> None:
    row = Row(
        pd.Series(data=[1, 2]),
        TableSchema(
            {
                "col1": RealNumber(),
                "col2": RealNumber(),
            }
        ),
    )
    assert row.get_column_names() == ["col1", "col2"]


def test_get_column_names_empty() -> None:
    row = Row(pd.Series(data=[]), TableSchema({}))
    assert not row.get_column_names()
