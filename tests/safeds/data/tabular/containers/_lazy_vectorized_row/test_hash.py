from collections.abc import Callable

import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    "table_factory",
    [
        lambda: Table({}),
        lambda: Table({"col1": []}),
        lambda: Table({"col1": [1, 2]}),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, table_factory: Callable[[], Table]) -> None:
        row_1 = _LazyVectorizedRow(table_factory())
        row_2 = _LazyVectorizedRow(table_factory())
        assert hash(row_1) == hash(row_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        table_factory: Callable[[], Table],
        snapshot: SnapshotAssertion,
    ) -> None:
        row = _LazyVectorizedRow(table_factory())
        assert hash(row) == snapshot


@pytest.mark.parametrize(
    ("table_1", "table_2"),
    [
        # too few columns
        (
            Table({"col1": [1]}),
            Table({}),
        ),
        # too many columns
        (
            Table({}),
            Table({"col1": [1]}),
        ),
        # different column order
        (
            Table({"col1": [1], "col2": [2]}),
            Table({"col2": [2], "col1": [1]}),
        ),
        # different column names
        (
            Table({"col1": [1]}),
            Table({"col2": [1]}),
        ),
        # different types
        (
            Table({"col1": [1]}),
            Table({"col1": ["1"]}),
        ),
        # too few rows
        (
            Table({"col1": [1, 2]}),
            Table({"col1": [1]}),  # Needs at least one value, so the types match
        ),
        # too many rows
        (
            Table({"col1": [1]}),  # Needs at least one value, so the types match
            Table({"col1": [1, 2]}),
        ),
    ],
    ids=[
        "too few columns",
        "too many columns",
        "different column order",
        "different column names",
        "different types",
        "too few rows",
        "too many rows",
    ],
)
def test_should_be_good_hash(table_1: Table, table_2: Table) -> None:
    row_1 = _LazyVectorizedRow(table_1)
    row_2 = _LazyVectorizedRow(table_2)
    assert hash(row_1) != hash(row_2)
