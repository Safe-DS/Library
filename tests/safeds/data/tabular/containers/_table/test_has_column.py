import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    [
        (Table({}), "C", False),
        (Table({"A": []}), "A", True),
        (Table({"A": []}), "B", False),
    ],
    ids=["empty", "has column", "doesn't have column"],
)
def test_should_return_if_column_is_in_table(table: Table, column: str, expected: bool) -> None:
    assert table.has_column(column) == expected
