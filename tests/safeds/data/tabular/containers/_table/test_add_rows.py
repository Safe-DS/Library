import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Row, Table
from safeds.exceptions import SchemaMismatchError


@pytest.mark.parametrize(
    ("table1", "rows", "table2"),
    [
        (
            Table({"col1": ["a", "b", "c"], "col2": [1, 2, 4]}),
            [Row({"col1": "d", "col2": 6}), Row({"col1": "e", "col2": 8})],
            Table({"col1": ["a", "b", "c", "d", "e"], "col2": [1, 2, 4, 6, 8]}),
        ),
        (
            Table(),
            [Row({"col1": "d", "col2": 6}), Row({"col1": "e", "col2": 8})],
            Table({"col1": ["d", "e"], "col2": [6, 8]}),
        ),
    ],
    ids=["Rows with string and integer values", "empty"],
)
def test_should_add_rows(table1: Table, rows: list[Row], table2: Table) -> None:
    table1 = table1.add_rows(rows)
    assert table1.schema == table2.schema
    assert table1 == table2


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [5, 7], "col2": [6, 8]}),
            Table({"col1": [1, 2, 1, 5, 7], "col2": [1, 2, 4, 6, 8]}),
        ),
        (
            Table({"col1": [2], "yikes": [5]}),
            Table(),
            Table({"col1": [2], "yikes": [5]}),
        ),
        (
            Table(),
            Table({"col1": [2], "yikes": [5]}),
            Table({"col1": [2], "yikes": [5]}),
        ),
        (
            Table({"col1": [], "yikes": []}),
            Table({"col1": [], "yikes": []}),
            Table({"col1": [], "yikes": []}),
        ),
    ],
    ids=["Rows from table", "add empty to table", "add on empty table", "rowless"],
)
def test_should_add_rows_from_table(table1: Table, table2: Table, expected: Table) -> None:
    table1 = table1.add_rows(table2)
    assert table1.schema == expected.schema
    assert table1 == expected


def test_should_raise_error_if_row_schema_invalid() -> None:
    table1 = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = [Row({"col1": 2, "col2": 4}), Row({"col1": 5, "col2": "Hallo"})]
    with pytest.raises(SchemaMismatchError, match=r"Failed because at least two schemas didn't match."):
        table1.add_rows(row)


def test_should_raise_schema_mismatch() -> None:
    with raises(SchemaMismatchError, match=r"Failed because at least two schemas didn't match."):
        Table({"a": [], "b": []}).add_rows([Row({"a": None, "b": None}), Row({"beer": None, "rips": None})])
    with raises(SchemaMismatchError, match=r"Failed because at least two schemas didn't match."):
        Table({"a": [], "b": []}).add_rows([Row({"beer": None, "rips": None}), Row({"a": None, "b": None})])
