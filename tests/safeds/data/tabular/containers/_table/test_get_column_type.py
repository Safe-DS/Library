import polars as pl
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType
from safeds.data.tabular.typing._polars_data_type import _PolarsColumnType
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "name", "expected"),
    [
        (
            Table({"col1": [1]}),
            "col1",
            _PolarsColumnType(pl.Int64()),
        ),
        (
            Table({"col1": ["a"]}),
            "col1",
            _PolarsColumnType(pl.String()),
        ),
    ],
    ids=["int column", "string column"],
)
def test_should_return_data_type_of_column(table: Table, name: str, expected: ColumnType) -> None:
    assert table.get_column_type(name) == expected


def test_should_raise_if_column_name_is_unknown() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).get_column_type("col1")
