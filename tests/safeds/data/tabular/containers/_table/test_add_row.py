import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.exceptions import SchemaMismatchError


@pytest.mark.parametrize(
    ("table", "row", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col1": 5, "col2": 6}),
            Table({"col1": [1, 2, 1, 5], "col2": [1, 2, 4, 6]}),
        ),
        (Table({"col2": [], "col4": []}), Row({"col2": 5, "col4": 6}), Table({"col2": [5], "col4": [6]})),
        (Table(), Row({"col2": 5, "col4": 6}), Table({"col2": [5], "col4": [6]})),
    ],
    ids=["added row", "added row to empty column", "empty row to empty table"],
)
def test_should_add_row(table: Table, row: Row, expected: Table) -> None:
    table = table.add_row(row)
    assert table == expected


def test_should_raise_error_if_row_schema_invalid() -> None:
    table1 = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = Row({"col1": 5, "col2": "Hallo"})
    with raises(SchemaMismatchError):
        table1.add_row(row)
