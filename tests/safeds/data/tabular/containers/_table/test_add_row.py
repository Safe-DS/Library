from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import SchemaMismatchError
from safeds.data.tabular.typing import Integer, Schema, String


def test_add_row_valid() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = Row([5, 6], table1.schema)
    table1 = table1.add_row(row)
    assert table1.count_rows() == 4
    assert table1.get_row(3) == row
    assert table1.schema == row.schema


def test_add_row_invalid() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = Row(
        [5, "Hallo"],
        Schema({"col1": Integer(), "col2": String()}),
    )
    with raises(SchemaMismatchError):
        table1 = table1.add_row(row)
