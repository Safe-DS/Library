import sys

import pytest
from safeds.data.tabular.typing import ColumnType


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
def test_should_size_be_greater_than_normal_object(type_: ColumnType) -> None:
    assert sys.getsizeof(type_) > sys.getsizeof(object())
