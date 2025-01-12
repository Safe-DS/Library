import pytest

from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (
            Schema({}),
            "Schema({})",
        ),
        (
            Schema({"col1": ColumnType.null()}),
            "Schema({'col1': null})",
        ),
        (
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            "Schema({\n    'col1': null,\n    'col2': int8\n})",
        ),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_return_a_string_representation(schema: Schema, expected: str) -> None:
    assert repr(schema) == expected
