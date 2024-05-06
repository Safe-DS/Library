import pytest
from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    argnames=("column", "expected"),
    argvalues=[
        (Column("A", []), Table({"A": []})),
        (Column("A", [1, 2, 3]), Table({"A": [1, 2, 3]})),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_a_table_with_this_column(column: Column, expected: Table) -> None:
    assert column.to_table() == expected
