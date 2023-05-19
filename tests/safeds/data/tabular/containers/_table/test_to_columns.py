import pytest
from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({"A": [54, 74], "B": [90, 2010]}), [Column("A", [54, 74]), Column("B", [90, 2010])]),
        (Table(), [])
    ],
    ids=["normal", "empty"]
)
def test_should_return_list_of_columns(table: Table, expected: list[Column]) -> None:
    assert table.to_columns() == expected
