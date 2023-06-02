import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import Schema, Integer, Anything
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "row", "expected", "expected_schema"),
    [
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Row({"col1": 5, "col2": 6}),
         Table({"col1": [1, 2, 1, 5], "col2": [1, 2, 4, 6]}),
         Schema({"col1": Integer(), "col2": Integer()})),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Row({"col1": "5", "col2": 6}),
         Table({"col1": [1, 2, 1, "5"], "col2": [1, 2, 4, 6]}),
         Schema({"col1": Anything(), "col2": Integer()})),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Table.from_rows([Row({"col1": "5", "col2": None}),
                          Row({"col1": "5", "col2": 2})]).get_row(0),
         Table({"col1": [1, 2, 1, "5"], "col2": [1, 2, 4, None]}),
         Schema({"col1": Anything(), "col2": Integer(False)})),
    ],
    ids=["added row", "different schemas", "different schemas and nullable"],
)
def test_should_add_row(table: Table, row: Row, expected: Table, expected_schema: Schema) -> None:
    table = table.add_row(row)
    assert table.number_of_rows == 4
    assert table.schema == expected_schema
    assert table == expected


@pytest.mark.parametrize(
    ("table", "row", "expected_error_msg"),
    [(Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Row({"col1": 5, "col3": "Hallo"}), r"Could not find column\(s\) 'col2'")],
    ids=["unknown column col2 in row"]
)
def test_should_raise_error_if_row_column_names_invalid(table, row, expected_error_msg) -> None:
    with raises(UnknownColumnNameError, match=expected_error_msg):
        table.add_row(row)
