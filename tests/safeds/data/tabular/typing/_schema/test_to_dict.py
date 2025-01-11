import pytest
from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (
            Schema({}),
            {},
        ),
        (
            Schema({"col1": ColumnType.null()}),
            {"col1": ColumnType.null()},
        ),
        (
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            {"col1": ColumnType.null(), "col2": ColumnType.int8()},
        ),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_return_dictionary(schema: Schema, expected: dict[str, ColumnType]) -> None:
    assert schema.to_dict() == expected
