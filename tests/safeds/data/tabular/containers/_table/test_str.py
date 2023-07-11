import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            "   col1  col2\n0     1     1\n1     2     2\n2     1     4",
        ),
        (Table({"col1": [], "col2": []}), "Empty DataFrame\nColumns: [col1, col2]\nIndex: []"),
        (Table(), "Empty DataFrame\nColumns: []\nIndex: []"),
        (Table({"col1": [1], "col2": [1]}), "   col1  col2\n0     1     1"),
    ],
    ids=["multiple rows", "rowless table", "empty table", "one row"],
)
def test_should_return_a_string_representation(table: Table, expected: str) -> None:
    assert str(table) == expected
