import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import SchemaMismatchError


@pytest.mark.parametrize(
    ("table1", "rows", "table2"),
    [
        (
            Table({"col1": ["a", "b", "c"], "col2": [1, 2, 4]}),
            [Row({"col1": "d", "col2": 6}), Row({"col1": "e", "col2": 8})],
            Table({"col1": ["a", "b", "c", "d", "e"], "col2": [1, 2, 4, 6, 8]}),
        ),
    ],
    ids=["Rows with string and integer values"],
)
def test_should_add_rows(table1: Table, rows: list[Row], table2: Table) -> None:
    table1 = table1.add_rows(rows)
    assert table1 == table2


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [5, 7], "col2": [6, 8]}),
            Table({"col1": [1, 2, 1, 5, 7], "col2": [1, 2, 4, 6, 8]}),
        ),
    ],
    ids=["Rows from table"],
)
def test_should_add_rows_from_table(table1: Table, table2: Table, expected: Table) -> None:
    table1 = table1.add_rows(table2)
    assert table1 == expected


def test_should_raise_error_if_row_schema_invalid() -> None:
    table1 = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    row = [Row({"col1": 2, "col2": 4}), Row({"col1": 5, "col2": "Hallo"})]
    with pytest.raises(SchemaMismatchError):
        table1.add_rows(row)
