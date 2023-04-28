import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a", []), "Column('a', [])"),
        (Column("a", [0]), "Column('a', [0])"),
        (Column("a", [0, "1"]), "Column('a', [0, '1'])"),
    ],
    ids=[
        "empty",
        "one row",
        "multiple rows",
    ],
)
def test_should_return_a_string_representation(column: Column, expected: str) -> None:
    assert repr(column) == expected
