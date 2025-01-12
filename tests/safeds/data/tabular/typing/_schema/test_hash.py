from collections.abc import Callable

import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    "schema_factory",
    [
        lambda: Schema({}),
        lambda: Schema({"col1": ColumnType.null()}),
        lambda: Schema({"col1": ColumnType.null(), "col2": ColumnType.null()}),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, schema_factory: Callable[[], Schema]) -> None:
        schema_1 = schema_factory()
        schema_2 = schema_factory()
        assert hash(schema_1) == hash(schema_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        schema_factory: Callable[[], Schema],
        snapshot: SnapshotAssertion,
    ) -> None:
        schema_ = schema_factory()
        assert hash(schema_) == snapshot


@pytest.mark.parametrize(
    ("schema_1", "schema_2"),
    [
        # not equal (too few columns)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({}),
        ),
        # not equal (too many columns)
        (
            Schema({}),
            Schema({"col1": ColumnType.null()}),
        ),
        # not equal (different column order)
        (
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            Schema({"col2": ColumnType.int8(), "col1": ColumnType.null()}),
        ),
        # not equal (different column names)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({"col2": ColumnType.null()}),
        ),
        # not equal (different types)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({"col1": ColumnType.int8()}),
        ),
    ],
    ids=[
        "too few columns",
        "too many columns",
        "different column order",
        "different column names",
        "different types",
    ],
)
def test_should_be_good_hash(
    schema_1: Schema,
    schema_2: Schema,
) -> None:
    assert hash(schema_1) != hash(schema_2)
