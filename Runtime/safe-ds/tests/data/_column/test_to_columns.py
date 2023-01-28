import pandas as pd
import pytest
from safeds.data import Column, Table


@pytest.mark.parametrize(
    "values, name, index",
    [([1, 4], "A", 0), ([2, 5], "B", 1)],
)
def test_to_columns(values: list[int], name: str, index: int) -> None:
    table = Table.from_csv("tests/resources/test_column_table.csv")
    columns_list: list[Column] = table.to_columns()

    column_expected: Column = Column(pd.Series(values, name=name), name)

    assert column_expected == columns_list[index]
