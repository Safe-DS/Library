import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), "Empty DataFrame\nColumns: []\nIndex: []"),
        (Table({"col1": [0]}), "   col1\n0     0"),
        (Table({"col1": [0, "1"], "col2": ["a", "b"]}), "  col1 col2\n0    0    a\n1    1    b"),
    ],
    ids=[
        "empty table",
        "table with one row",
        "table with multiple rows",
    ],
)
def test_should_return_a_string_representation(table: Table, expected: str) -> None:
    assert repr(table) == expected
