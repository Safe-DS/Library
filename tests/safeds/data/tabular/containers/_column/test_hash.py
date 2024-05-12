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
    ("column1", "column2"),
    [
        (Column("a"), Column("a")),
        (Column("a", [1, 2, 3]), Column("a", [1, 2, 3])),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_same_hash_for_equal_columns(column1: Column, column2: Column) -> None:
    assert hash(column1) == hash(column2)


@pytest.mark.parametrize(
    ("column1", "column2"),
    [
        (Column("a"), Column("b")),
        (Column("a", [1, 2, 3]), Column("a", [1, 2])),
        (Column("a", [1, 2, 3]), Column("a", ["1", "2", "3"])),
        # We don't use the column values in the hash calculation
    ],
    ids=[
        "different names",
        "different lengths",
        "different types",
    ],
)
def test_should_ideally_return_different_hash_for_unequal_columns(column1: Column, column2: Column) -> None:
    assert hash(column1) != hash(column2)
