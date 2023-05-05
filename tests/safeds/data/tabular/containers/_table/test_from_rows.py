import pytest
from safeds.data.tabular.containers import Table, Row
from safeds.data.tabular.exceptions import MissingDataError, SchemaMismatchError
from safeds.data.tabular.typing import Integer, Schema, String


@pytest.mark.parametrize(
    "table",
    [
        (Table([[1, 4, "d"], [2, 5, "e"], [3, 6, "f"]],
               Schema({"A": Integer(), "B": Integer(), "D": String()}))),
    ],
    ids=["empty"]
)
def test_should_create_table_from_rows(table: Table) -> None:
    rows_is = table.to_rows()
    table_is = Table.from_rows(rows_is)

    assert table_is == table


def test_should_raise_error_if_data_missing() -> None:
    with pytest.raises(MissingDataError):
        Table.from_rows([])


def test_should_raise_error_if_mismatching_schema() -> None:
    rows = [
        Row({"A": 1, "B": 2}),
        Row({"A": 2, "B": "a"})
    ]
    with pytest.raises(SchemaMismatchError):
        Table.from_rows(rows)
