import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import SchemaMismatchError


@pytest.mark.parametrize(
    ("table", "row"),
    [
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Row({"col1": 5, "col2": 6})),
    ],
    ids=["added row"],
)
def test_should_add_row(table: Table, row: Row) -> None:
    table = table.add_row(row)
    assert table.number_of_rows == 4
    assert table.get_row(3) == row
    assert table.schema == row._schema


def test_should_raise_error_if_row_schema_invalid() -> None:
    table1 = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = Row({"col1": 5, "col2": "Hallo"})
    with raises(SchemaMismatchError):
        table1.add_row(row)
