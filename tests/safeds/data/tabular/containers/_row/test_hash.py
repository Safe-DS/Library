import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (Table(), Table({"A": ["a", "aa", "aaa"]}), False),
        (Table(), Table(), True),
        (Table({"A": ["a", "aa", "aaa"]}), Table({"A": ["a", "aa", "aaa"]}), True),
        (Table({"A": ["a", "aa", "aaa"]}), Table({"B": ["a", "aa", "aaa"]}), False),
    ],
    ids=[
        "empty and different table",
        "same empty tables",
        "same tables",
        "different tables",
    ],
)
def test_should_return_consistent_hashes(table1: Table, table2: Table, expected: bool) -> None:
    row1 = _LazyVectorizedRow(table=table1)
    row2 = _LazyVectorizedRow(table=table2)
    assert (hash(row1) == hash(row2)) == expected
