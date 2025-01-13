import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    [
        (Table({}), "C", False),
        (Table({"A": []}), "A", True),
        (Table({"A": []}), "B", False),
    ],
    ids=["empty", "has column", "doesn't have column"],
)
def test_should_return_if_column_is_in_row(table: Table, column: str, expected: bool) -> None:
    row = _LazyVectorizedRow(table)
    assert row.has_column(column) == expected
