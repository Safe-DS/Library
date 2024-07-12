import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), []),
        (Table({"A": ["a", "aa", "aaa"]}), ["A"]),
        (Table({"A": ["a", "aa", "aaa"], "B": ["b", "bb", "bbb"], "C": ["c", "cc", "ccc"]}), ["A", "B", "C"]),
    ],
    ids=[
        "empty",
        "one column",
        "three columns",
    ],
)
def test_should_return_same_list_of_column_name_with_iter(table: Table, expected: list) -> None:
    row: Row[any] = _LazyVectorizedRow(table=table)
    iterable = iter(row)
    assert list(iterable) == expected
