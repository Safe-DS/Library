from typing import Any

import pytest
from safeds.data.tabular.containers import Column, Row


@pytest.mark.parametrize(
    ("column1", "column2"),
    [
        (Column("a"), Column("a")),
        (Column("a", [1, 2, 3]), Column("a", [1, 2, 3])),
        (Column("a", [1, 2, 3]), Column("a", [1, 2, 4])),
    ],
    ids=[
        "empty columns",
        "equal columns",
        "different values",
    ],
)
def test_should_return_same_hash_for_equal_columns(column1: Column, column2: Column) -> None:
    assert hash(column1) == hash(column2)


@pytest.mark.parametrize(
    ("column1", "column2"),
    [
        (Column("a"), Column("b")),
        (Column("a", [1, 2, 3]), Column("a", ["1", "2", "3"])),
    ],
    ids=[
        "different names",
        "different types",
    ],
)
def test_should_return_different_hash_for_unequal_columns(column1: Column, column2: Column) -> None:
    assert hash(column1) != hash(column2)
