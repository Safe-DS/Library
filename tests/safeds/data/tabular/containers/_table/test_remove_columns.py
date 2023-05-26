import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table1", "expected", "columns"),
    [
        (Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table({"col1": [1, 2, 1]}), ["col2"]),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table({}), ["col1", "col2"]),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), []),
    ],
    ids=["one column", "multiple columns", "no columns"],
)
def test_should_remove_table_columns(table1: Table, expected: Table, columns: list[str]) -> None:
    table1 = table1.remove_columns(columns)
    assert table1.schema == expected.schema
    assert table1 == expected


def test_should_raise_if_column_not_found() -> None:
    table = Table({"A": [1], "B": [2]})
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'C'"):
        table.remove_columns(["C"])
