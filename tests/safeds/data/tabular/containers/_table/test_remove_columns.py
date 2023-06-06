import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table1", "expected", "columns"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}),
            Table({"col1": [1, 2, 1]}),
            ["col2"]
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({}),
            ["col1", "col2"]
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [1, 2, 1],"col2": [1, 2, 4]}),
            []
        ),
        (Table(),
         Table(),
         []),
    ],
    ids=["one column", "multiple columns", "no columns", "empty"],
)
def test_should_remove_table_columns(table1: Table, expected: Table, columns: list[str]) -> None:
    table1 = table1.remove_columns(columns)
    assert table1.schema == expected.schema
    assert table1 == expected


@pytest.mark.parametrize("table", [Table({"A": [1], "B": [2]}), Table()], ids=["normal", "empty"])
def test_should_raise_if_column_not_found(table: Table) -> None:
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'C'"):
        table.remove_columns(["C"])
