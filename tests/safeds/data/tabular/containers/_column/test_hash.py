from collections.abc import Callable

import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "column_factory",
    [
        lambda: Column("col1", []),
        lambda: Column("col1", [1, 2]),
    ],
    ids=[
        "no rows",
        "with data",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, column_factory: Callable[[], Column]) -> None:
        column_1 = column_factory()
        column_2 = column_factory()
        assert hash(column_1) == hash(column_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        column_factory: Callable[[], Column],
        snapshot: SnapshotAssertion,
    ) -> None:
        column = column_factory()
        assert hash(column) == snapshot


@pytest.mark.parametrize(
    ("column_1", "column_2"),
    [
        # different names
        (
            Column("col1", [1]),
            Column("col2", [1]),
        ),
        # different types
        (
            Column("col1", [1]),
            Column("col1", ["1"]),
        ),
        # too few rows
        (
            Column("col1", [1, 2]),
            Column("col1", [1]),  # Needs at least one value, so the types match
        ),
        # too many rows
        (
            Column("col1", [1]),  # Needs at least one value, so the types match
            Column("col1", [1, 2]),
        ),
    ],
    ids=[
        "different names",
        "different types",
        "too few rows",
        "too many rows",
    ],
)
def test_should_be_good_hash(column_1: Column, column_2: Column) -> None:
    assert hash(column_1) != hash(column_2)
