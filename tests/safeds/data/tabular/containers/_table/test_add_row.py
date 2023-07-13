import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import Anything, Integer, Schema
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "row", "expected", "expected_schema"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col1": 5, "col2": 6}),
            Table({"col1": [1, 2, 1, 5], "col2": [1, 2, 4, 6]}),
            Schema({"col1": Integer(), "col2": Integer()}),
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col1": "5", "col2": 6}),
            Table({"col1": [1, 2, 1, "5"], "col2": [1, 2, 4, 6]}),
            Schema({"col1": Anything(), "col2": Integer()}),
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col1": "5", "col2": None}),
            Table({"col1": [1, 2, 1, "5"], "col2": [1, 2, 4, None]}),
            Schema({"col1": Anything(), "col2": Integer(is_nullable=True)}),
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col1": 5, "col2": 6}),
            Table({"col1": [1, 2, 1, 5], "col2": [1, 2, 4, 6]}),
            Schema({"col1": Integer(), "col2": Integer()}),
        ),
        (
            Table({"col1": [], "col2": []}),
            Row({"col1": 5, "col2": 6}),
            Table({"col1": [5], "col2": [6]}),
            Schema({"col1": Integer(), "col2": Integer()}),
        ),
    ],
    ids=[
        "added row",
        "different schemas",
        "different schemas and nullable",
        "add row to rowless table",
        "add row to empty table",
    ],
)
def test_should_add_row(table: Table, row: Row, expected: Table, expected_schema: Schema) -> None:
    result = table.add_row(row)
    assert result.number_of_rows - 1 == table.number_of_rows
    assert result.schema == expected_schema
    assert result == expected


@pytest.mark.parametrize(
    ("table", "row", "expected_error_msg"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col1": 5, "col3": "Hallo"}),
            r"Could not find column\(s\) 'col2'",
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Row({"col3": 5, "col4": "Hallo"}),
            r"Could not find column\(s\) 'col1, col2'",
        ),
    ],
    ids=["unknown column col2 in row", "multiple columns missing"],
)
def test_should_raise_error_if_row_column_names_invalid(table: Table, row: Row, expected_error_msg: str) -> None:
    with raises(UnknownColumnNameError, match=expected_error_msg):
        table.add_row(row)
