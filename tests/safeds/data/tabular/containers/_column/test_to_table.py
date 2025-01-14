import pytest

from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("col1", []), Table({"col1": []})),
        (Column("col1", [1, 2, 3]), Table({"col1": [1, 2, 3]})),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_table_with_this_column(column: Column, expected: Table) -> None:
    assert column.to_table() == expected
