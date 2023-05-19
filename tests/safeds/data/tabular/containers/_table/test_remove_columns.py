import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table({"col1": [1, 2, 1]})),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table({"col1": [1, 2, 1]})),
    ],
    ids=["String", "Integer"],
)
def test_should_remove_table_columns(table1: Table, expected: Table) -> None:
    table1 = table1.remove_columns(["col2"])
    assert table1 == expected

@pytest.mark.parametrize(
    "table",
    [
        Table({"A": [1], "B": [2]}),
        Table()
    ]
)
def test_should_raise_if_column_not_found(table) -> None:
    with pytest.raises(UnknownColumnNameError):
        table.remove_columns(["C"])
