import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import DuplicateColumnError

# TODO: merge into add_columns file


@pytest.mark.parametrize(
    ("table1", "column", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Column("col3", ["a", "b", "c"]),
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": ["a", "b", "c"]}),
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Column("col3", [0, -1, -2]),
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2]}),
        ),
        (
            Table(),
            Column("col3", []),
            Table({"col3": []}),
        ),
        (
            Table(),
            Column("col3", [1]),
            Table({"col3": [1]}),
        ),
    ],
    ids=["String", "Integer", "empty with empty column", "empty with filled column"],
)
def test_should_add_column(table1: Table, column: Column, expected: Table) -> None:
    table1 = table1.add_columns(column)
    # assert table1.schema == expected.schema
    assert table1 == expected


def test_should_raise_error_if_column_name_exists() -> None:
    table1 = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(DuplicateColumnError):
        table1.add_columns(Column("col1", ["a", "b", "c"]))


# TODO
# def test_should_raise_error_if_column_size_invalid() -> None:
#     table1 = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
#     with pytest.raises(ColumnSizeError, match=r"Expected a column of size 3 but got column of size 4."):
#         table1.add_columns(Column("col3", ["a", "b", "c", "d"]))
