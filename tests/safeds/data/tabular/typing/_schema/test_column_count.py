import pytest

from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (Schema({}), 0),
        (Schema({"col1": ColumnType.null()}), 1),
        (Schema({"col1": ColumnType.null(), "col2": ColumnType.null()}), 2),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_return_number_of_columns(schema: Schema, expected: int) -> None:
    assert schema.column_count == expected
