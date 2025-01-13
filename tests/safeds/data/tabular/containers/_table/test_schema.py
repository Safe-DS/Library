import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table({}),
            Schema({}),
        ),
        (
            Table({"col1": []}),
            Schema({"col1": ColumnType.null()}),
        ),
        (
            Table({"col1": [1], "col2": [1]}),
            Schema({"col1": ColumnType.int64(), "col2": ColumnType.int64()}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
def test_should_return_schema(table: Table, expected: Schema) -> None:
    assert table.schema == expected
