import pytest

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("type_", "expected"),
    [
        (ColumnType.float32(), False),
        (ColumnType.float64(), False),
        (ColumnType.int8(), True),
        (ColumnType.int16(), True),
        (ColumnType.int32(), True),
        (ColumnType.int64(), True),
        (ColumnType.uint8(), True),
        (ColumnType.uint16(), True),
        (ColumnType.uint32(), True),
        (ColumnType.uint64(), True),
        (ColumnType.date(), False),
        (ColumnType.datetime(), False),
        (ColumnType.duration(), False),
        (ColumnType.time(), False),
        (ColumnType.string(), False),
        (ColumnType.binary(), False),
        (ColumnType.boolean(), False),
        (ColumnType.null(), False),
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
def test_should_return_whether_type_represents_ints(type_: ColumnType, expected: bool) -> None:
    assert type_.is_int == expected
