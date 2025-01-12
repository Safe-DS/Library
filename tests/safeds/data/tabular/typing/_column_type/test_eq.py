from typing import Any

import pytest

from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("type_1", "type_2", "expected"),
    [
        (ColumnType.float32(), ColumnType.float32(), True),
        (ColumnType.float32(), ColumnType.float64(), False),
        (ColumnType.float32(), ColumnType.int32(), False),
        (ColumnType.int32(), ColumnType.uint32(), False),
        (ColumnType.int32(), ColumnType.string(), False),
    ],
    ids=[
        "equal",
        "not equal (different bit count)",
        "not equal (float vs. int)",
        "not equal (signed vs. unsigned)",
        "not equal (numeric vs. non-numeric)",
    ],
)
def test_should_return_whether_column_types_are_equal(type_1: Table, type_2: Table, expected: bool) -> None:
    assert (type_1.__eq__(type_2)) == expected


@pytest.mark.parametrize(
    "type_",
    [
        ColumnType.float32(),
        ColumnType.float64(),
        ColumnType.int8(),
        ColumnType.int16(),
        ColumnType.int32(),
        ColumnType.int64(),
        ColumnType.uint8(),
        ColumnType.uint16(),
        ColumnType.uint32(),
        ColumnType.uint64(),
        ColumnType.date(),
        ColumnType.datetime(),
        ColumnType.duration(),
        ColumnType.time(),
        ColumnType.string(),
        ColumnType.binary(),
        ColumnType.boolean(),
        ColumnType.null(),
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
def test_should_return_true_if_column_types_are_identical(type_: ColumnType) -> None:
    assert (type_.__eq__(type_)) is True


@pytest.mark.parametrize(
    ("type_", "other"),
    [
        (ColumnType.null(), None),
        (ColumnType.null(), Column("col1", [])),
    ],
    ids=[
        "ColumnType vs. None",
        "ColumnType vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_is_not_column_type(type_: ColumnType, other: Any) -> None:
    assert (type_.__eq__(other)) is NotImplemented
