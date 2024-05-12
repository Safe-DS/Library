import pytest
from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("table1", "columns", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            [Column("col3", [0, -1, -2]), Column("col4", ["a", "b", "c"])],
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
        ),
        (
            Table(),
            [Column("col3", []), Column("col4", [])],
            Table({"col3": [], "col4": []}),
        ),
        (
            Table(),
            [Column("col3", [1]), Column("col4", [2])],
            Table({"col3": [1], "col4": [2]}),
        ),
    ],
    ids=["add 2 columns", "empty with empty column", "empty with filled column"],
)
def test_should_add_columns(table1: Table, columns: list[Column], expected: Table) -> None:
    table1 = table1.add_columns(columns)
    # assert table1.schema == expected.schema
    assert table1 == expected


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
        ),
        (Table(), Table({"col1": [1, 2], "col2": [60, 2]}), Table({"col1": [1, 2], "col2": [60, 2]})),
        (
            Table({"col1": [1, 2], "col2": [60, 2]}),
            Table(),
            Table({"col1": [1, 2], "col2": [60, 2]}),
        ),
        (Table({"yeet": [], "col": []}), Table({"gg": []}), Table({"yeet": [], "col": [], "gg": []})),
    ],
    ids=["add a table with 2 columns", "empty add filled", "filled add empty", "rowless"],
)
def test_should_add_columns_from_table(table1: Table, table2: Table, expected: Table) -> None:
    table1 = table1.add_table_as_columns(table2)  # TODO: move to separate test file
    assert table1.schema == expected.schema
    assert table1 == expected


#  TODO - separate test for add_table_as_columns and a new one here
# @pytest.mark.parametrize(
#     ("table", "columns", "error_message_regex"),
#     [
#         (
#             Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
#             [Column("col3", ["a", "b", "c", "d"]), Column("col4", ["e", "f", "g", "h"])],
#             r"Expected a column of size 3 but got column of size 4.",
#         ),
#     ],
#     ids=["Two Columns with too many values"],
# )
# def test_should_raise_error_if_column_size_invalid(
#     table: Table,
#     columns: list[Column] | Table,
#     error_message_regex: str,
# ) -> None:
#     with pytest.raises(ColumnSizeError, match=error_message_regex):
#         table.add_columns(columns)

#  TODO - separate test for add_table_as_columns and a new one here
# @pytest.mark.parametrize(
#     ("table", "columns", "error_message_regex"),
#     [
#         (
#             Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
#             [Column("col2", ["a", "b", "c"]), Column("col3", [2, 3, 4])],
#             r"Column 'col2' already exists.",
#         ),
#     ],
#     ids=["Column already exists"],
# )
# def test_should_raise_error_if_column_name_in_result_column(
#     table: Table,
#     columns: list[Column] | Table,
#     error_message_regex: str,
# ) -> None:
#     with pytest.raises(DuplicateColumnNameError, match=error_message_regex):
#         table.add_columns(columns)
