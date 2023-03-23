import pandas as pd
import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import MissingDataError


def test_from_columns() -> None:
    table_expected = Table.from_csv(resolve_resource_path("test_column_table.csv"))
    columns_table: list[Column] = [
        Column(pd.Series([1, 4]), "A"),
        Column(pd.Series([2, 5]), "B"),
    ]
    table_restored: Table = Table.from_columns(columns_table)

    assert table_restored == table_expected


def test_from_columns_invalid() -> None:
    with pytest.raises(MissingDataError):
        Table.from_columns([])
