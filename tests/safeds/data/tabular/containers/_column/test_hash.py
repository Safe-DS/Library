import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a"), 1581717131331298536),
        (Column("a", [1, 2, 3]), 239695622656180157),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_be_deterministic(column: Column, expected: int) -> None:
    assert hash(column) == expected


@pytest.mark.parametrize(
    ("column1", "column2", "expected"),
    [
        (Column("a"), Column("a"), True),
        (Column("a", [1, 2, 3]), Column("a", [1, 2, 3]), True),
        (Column("a"), Column("b"), False),
        (Column("a", [1, 2, 3]), Column("a", [1, 2]), False),
        (Column("a", [1, 2, 3]), Column("a", ["1", "2", "3"]), False),
        # We don't use the column values in the hash calculation
    ],
    ids=[
        "equal empty",
        "equal non-empty",
        "different names",
        "different lengths",
        "different types",
    ],
)
def test_should_be_good_hash(column1: Column, column2: Column, expected: bool) -> None:
    assert (hash(column1) == hash(column2)) == expected
