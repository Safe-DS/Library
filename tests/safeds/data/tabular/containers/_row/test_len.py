import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), 0),
        (Table({"A": ["a", "aa", "aaa"]}), 1),
        (Table({"A": ["a", "aa", "aaa"], "B": ["b", "bb", "bbb"]}), 2),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_have_same_length_as_number_of_columns(table: Table, expected: int) -> None:
    row = _LazyVectorizedRow(table=table)
    assert len(row) == expected
