import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "name", "expected"),
    [
        (Table({"col1": [1]}), "col1", Column("col1", [1])),
        (Table({"col1": [1], "col2": [2]}), "col2", Column("col2", [2])),
    ],
    ids=[
        "one column",
        "multiple columns",
    ],
)
def test_should_get_column(table: Table, name: str, expected: Column) -> None:
    assert table.get_column(name) == expected


def test_should_raise_if_column_name_is_unknown() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).get_column("col1")
