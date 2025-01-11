import pytest
from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (
            Schema({}),
            "Empty schema",
        ),
        (
            Schema({"col1": ColumnType.null()}),
            "| Column Name | Column Type |\n| --- | --- |\n| col1 | Null |",
        ),
        (
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            "| Column Name | Column Type |\n| --- | --- |\n| col1 | Null |\n| col2 | Int8 |",
        ),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_return_markdown(schema: Schema, expected: str) -> None:
    assert schema._repr_markdown_() == expected
