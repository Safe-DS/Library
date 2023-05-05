import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table.from_dict({"col1": [1, 2, 1]})),
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table.from_dict({"col1": [1, 2, 1]})),
    ],
    ids=["String", "Integer"],
)
def test_should_remove_table_columns(table1: Table, expected: Table) -> None:
    table1 = table1.remove_columns(["col2"])
    assert table1 == expected


def test_should_raise_if_column_not_found() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(UnknownColumnNameError):
        table.remove_columns(["C"])
