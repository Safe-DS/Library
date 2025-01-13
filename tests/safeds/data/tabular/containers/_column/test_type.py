import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (
            Column("col1", [1]),
            ColumnType.int64(),
        ),
        (
            Column("col1", ["a"]),
            ColumnType.string(),
        ),
    ],
    ids=[
        "int column",
        "string column",
    ],
)
def test_should_return_type(column: Column, expected: ColumnType) -> None:
    assert column.type == expected
