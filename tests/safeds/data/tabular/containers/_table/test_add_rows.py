import pytest
from safeds.data.tabular.containers import Row, Table

@pytest.mark.parametrize(
    ("table1", "row1", "row2", "table2"),
    [
        (Table.from_dict({"col1": ["a", "b", "c"], "col2": [1, 2, 4]}),
         Row({"col1": "d", "col2": 6}),
         Row({"col1": "e", "col2": 8}),
         Table.from_dict({"col1": ["a", "b","c", "d", "e"], "col2": [1, 2, 4, 6, 8]})),
    ],
    ids=["Rows with string and integer values"],
)
def test_should_add_rows(table1: Table, row1: Row, row2: Row, table2: Table) -> None:
    table1 = table1.add_rows([row1, row2])
    assert table1 == table2

@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Table.from_dict({"col1": [5, 7], "col2": [6, 8]}),
         Table.from_dict({"col1": [1, 2, 1, 5, 7], "col2": [1, 2, 4, 6, 8]})),
    ],
    ids=["Rows from table"],
)
def test_should_add_rows_from_table(table1: Table, table2: Table, expected: Table) -> None:
    table1 = table1.add_rows(table2)
    assert table1 == expected
