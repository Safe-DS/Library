import pandas as pd
from safeds.data.tabular.containers import Column, Table

from tests.helpers import resolve_resource_path


def test_from_columns() -> None:
    table_expected = Table.from_csv_file(resolve_resource_path("test_column_table.csv"))
    columns_table: list[Column] = [
        Column("A", pd.Series([1, 4])),
        Column("B", pd.Series([2, 5])),
    ]
    table_restored: Table = Table.from_columns(columns_table)

    assert table_restored == table_expected
