import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a", []), False),
        (Column("a", [0]), True),
        (Column("a", [0.5]), True),
        (Column("a", [0, None]), True),
        (Column("a", ["a", "b"]), False),
    ],
    ids=[
        "empty",
        "int",
        "float",
        "numeric with missing",
        "non-numeric",
    ],
)
def test_should_return_whether_column_is_numeric(column: Column, expected: bool) -> None:
    assert column.is_numeric == expected
