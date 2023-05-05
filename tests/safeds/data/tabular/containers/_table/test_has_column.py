import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    [(Table.from_dict({"A": [1], "B": [2]}), "A", True), (Table.from_dict({"A": [1], "B": [2]}), "C", False)],
    ids=["has column", "doesn't have column"],
)
def test_should_return_if_column_is_in_table(table: Table, column: str, expected: bool) -> None:
    assert table.has_column(column) == expected
