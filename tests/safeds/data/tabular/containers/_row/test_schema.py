import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table"),
    [
        (Table()),
        (Table({"A": ["a", "aa", "aaa"]})),
        (Table({"A": ["a", "aa", "aaa"], "B": ["b", "bb", "bbb"]})),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_return_same_schema(table: Table) -> None:
    row = _LazyVectorizedRow(table=table)
    assert table.schema == row.schema
