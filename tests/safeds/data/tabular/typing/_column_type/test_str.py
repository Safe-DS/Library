import pytest
from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("type_", "expected"),
    [
        (ColumnType.float32(), "float32"),
        (ColumnType.float64(), "float64"),
        (ColumnType.int8(), "int8"),
        (ColumnType.int16(), "int16"),
        (ColumnType.int32(), "int32"),
        (ColumnType.int64(), "int64"),
        (ColumnType.uint8(), "uint8"),
        (ColumnType.uint16(), "uint16"),
        (ColumnType.uint32(), "uint32"),
        (ColumnType.uint64(), "uint64"),
        (ColumnType.date(), "date"),
        (ColumnType.datetime(), "datetime"),
        (ColumnType.duration(), "duration"),
        (ColumnType.time(), "time"),
        (ColumnType.string(), "string"),
        (ColumnType.binary(), "binary"),
        (ColumnType.boolean(), "boolean"),
        (ColumnType.null(), "null"),
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
def test_should_return_a_string_representation(type_: ColumnType, expected: bool) -> None:
    assert str(type_) == expected
