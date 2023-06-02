import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.exceptions import UnknownColumnNameError


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


@pytest.mark.parametrize(
    ("table", "rows", "expected_error_msg"),
    [
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         [Row({"col1": 2, "col3": 4}), Row({"col1": 5, "col2": "Hallo"})],
         r"aa"
         ),
    ]
)
def test_should_raise_error_if_row_column_names_invalid(table: Table, rows: list[Row], expected_error_msg: str) -> None:
    with pytest.raises(UnknownColumnNameError, match=expected_error_msg):
        table.add_rows(rows)
