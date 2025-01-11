from collections.abc import Callable

import pytest
from safeds.data.tabular.typing import ColumnType
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    "type_factory",
    [
        lambda: ColumnType.float32(),
        lambda: ColumnType.float64(),
        lambda: ColumnType.int8(),
        lambda: ColumnType.int16(),
        lambda: ColumnType.int32(),
        lambda: ColumnType.int64(),
        lambda: ColumnType.uint8(),
        lambda: ColumnType.uint16(),
        lambda: ColumnType.uint32(),
        lambda: ColumnType.uint64(),
        lambda: ColumnType.date(),
        lambda: ColumnType.datetime(),
        lambda: ColumnType.duration(),
        lambda: ColumnType.time(),
        lambda: ColumnType.string(),
        lambda: ColumnType.binary(),
        lambda: ColumnType.boolean(),
        lambda: ColumnType.null(),
    ],
    ids=[
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "date",
        "datetime",
        "duration",
        "time",
        "string",
        "binary",
        "boolean",
        "null",
    ],
)
class TestSameHash:
    def test_should_return_same_hash_for_equal_column_types(self, type_factory: Callable[[], ColumnType]) -> None:
        type_1 = type_factory()
        type_2 = type_factory()
        assert hash(type_1) == hash(type_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        type_factory: Callable[[], ColumnType],
        snapshot: SnapshotAssertion,
    ) -> None:
        type_ = type_factory()
        assert hash(type_) == snapshot


@pytest.mark.parametrize(
    ("type_1", "type_2"),
    [
        (ColumnType.float32(), ColumnType.float64()),
        (ColumnType.float32(), ColumnType.int32()),
        (ColumnType.int32(), ColumnType.uint32()),
        (ColumnType.int32(), ColumnType.string()),
    ],
    ids=[
        "different bit count",
        "float vs. int",
        "signed vs. unsigned",
        "numeric vs. non-numeric",
    ],
)
def test_should_return_different_hash_for_unequal_column_types(
    type_1: ColumnType,
    type_2: ColumnType,
) -> None:
    assert hash(type_1) != hash(type_2)
