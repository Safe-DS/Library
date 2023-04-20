from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import SchemaMismatchError


def test_add_row_valid() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = Row({"col1": 5, "col2": 6})
    table1 = table1.add_row(row)
    assert table1.n_rows == 4
    assert table1.get_row(3) == row
    assert table1.schema == row._schema


def test_add_row_invalid() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = Row({"col1": 5, "col2": "Hallo"})
    with raises(SchemaMismatchError):
        table1 = table1.add_row(row)
