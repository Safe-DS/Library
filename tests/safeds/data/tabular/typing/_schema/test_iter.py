import pytest

from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (Schema({}), []),
        (Schema({"col1": ColumnType.null()}), ["col1"]),
        (Schema({"col1": ColumnType.null(), "col2": ColumnType.null()}), ["col1", "col2"]),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_return_column_names(schema: Schema, expected: list[str]) -> None:
    assert list(schema) == expected
