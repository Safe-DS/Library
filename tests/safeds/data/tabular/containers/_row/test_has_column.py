import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "column_name", "expected"),
    [
        (Table({}), "A", False),
        (Table({"A": ["a", "aa", "aaa"]}), "A", True),
        (Table({"A": ["a", "aa", "aaa"]}), "B", False),
    ],
    ids=[
        "empty table",
        "table with existing column_name",
        "table with non existing column_name",
    ],
)
def test_should_have_column_name(table: Table, column_name: str, expected: bool) -> None:
    row = _LazyVectorizedRow(table=table)
    assert row.has_column(column_name) == expected
