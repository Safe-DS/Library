import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({"a": [1, 2], "b": [2, 3], "c": [None, 4]}), ["c"]),
        (Table({"a": [1, 2], "b": [2, None], "c": [None, 4]}), ["b", "c"]),
        (Table({"a": [None, 2], "b": [2, None], "c": [None, 4]}), ["a", "b", "c"]),
        (Table({"a": [1, 2], "b": [2, 3], "c": [3, 4]}), []),
    ],
    ids=["one column", "multiple columns", "all columns", "no columns"],
)
def test_should_list_columns_with_missing_values(table: Table, expected: list[str]) -> None:
    assert table.list_columns_with_missing_values() == expected
