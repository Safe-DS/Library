import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import ColumnSizeError, DuplicateColumnNameError


@pytest.mark.parametrize(
    ("table1", "column1", "column2", "expected"),
    [
        (
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Column("col3", [0, -1, -2]),
            Column("col4", ["a", "b", "c"]),
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
        ),
    ],
    ids=["add 2 columns"],
)
def test_should_add_columns(table1: Table, column1: Column, column2: Column, expected: Table) -> None:
    table1 = table1.add_columns([column1, column2])
    assert table1 == expected


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table.from_dict({"col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
        ),
    ],
    ids=["add a table with 2 columns"],
)
def test_should_add_columns_from_table(table1: Table, table2: Table, expected: Table) -> None:
    table1 = table1.add_columns(table2)
    assert table1 == expected


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            [Column("col3", ["a", "b", "c", "d"]), Column("col4", ["e", "f", "g", "h"])],
        ),
    ],
    ids=["Two Columns with too many values"],
)
def test_should_raise_error_if_column_size_invalid(table: Table, columns: list[Column] | Table) -> None:
    with pytest.raises(ColumnSizeError):
        table = table.add_columns(columns)


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            [Column("col2", ["a", "b", "c"]), Column("col3", [2, 3, 4])],
        ),
    ],
    ids=["Column already exists"],
)
def test_should_raise_error_if_column_name_in_result_column(table: Table, columns: list[Column] | Table) -> None:
    with pytest.raises(DuplicateColumnNameError):
        table = table.add_columns(columns)
