import pytest
from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("type_", "expected"),
    [
        (ColumnType.float32(), False),
        (ColumnType.float64(), False),
        (ColumnType.int8(), False),
        (ColumnType.int16(), False),
        (ColumnType.int32(), False),
        (ColumnType.int64(), False),
        (ColumnType.uint8(), False),
        (ColumnType.uint16(), False),
        (ColumnType.uint32(), False),
        (ColumnType.uint64(), False),
        (ColumnType.date(), True),
        (ColumnType.datetime(), True),
        (ColumnType.duration(), True),
        (ColumnType.time(), True),
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
def test_should_return_whether_type_represents_temporal_data(type_: ColumnType, expected: bool) -> None:
    assert type_.is_temporal == expected
