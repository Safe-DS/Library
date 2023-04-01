import pandas as pd
import pytest
from safeds.data.tabular.containers import Column, Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("values", "name", "index"),
    [([1, 4], "A", 0), ([2, 5], "B", 1)],
)
def test_to_columns(values: list[int], name: str, index: int) -> None:
    table = Table.from_csv_file(resolve_resource_path("test_column_table.csv"))
    columns_list: list[Column] = table.to_columns()

    column_expected: Column = Column(name, pd.Series(values, name=name))

    assert column_expected == columns_list[index]
