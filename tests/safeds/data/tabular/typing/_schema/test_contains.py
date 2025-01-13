import pytest

from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema", "column", "expected"),
    [
        (
            Schema({}),
            "C",
            False,
        ),
        (
            Schema({"A": ColumnType.null()}),
            "A",
            True,
        ),
        (
            Schema({"A": ColumnType.null()}),
            "B",
            False,
        ),
        (
            Schema({"A": ColumnType.null()}),
            1,
            False,
        ),
    ],
    ids=[
        "empty",
        "has column",
        "doesn't have column",
        "key is not string",
    ],
)
def test_should_check_if_column_is_in_schema(schema: Schema, column: str, expected: bool) -> None:
    assert (column in schema) == expected
