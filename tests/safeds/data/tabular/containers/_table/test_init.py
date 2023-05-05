import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import ColumnLengthMismatchError
from safeds.data.tabular.typing import Integer, Schema


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), Schema({})),
        (Table({}), Schema({})),
        (Table({"col1": [0]}), Schema({"col1": Integer()})),
    ],
    ids=[
        "empty",
        "empty (explicit)",
        "one column",
    ],
)
def test_should_infer_the_schema(table: Table, expected: Schema) -> None:
    assert table.schema == expected


def test_should_raise_error_if_columns_have_different_lengths() -> None:
    with pytest.raises(ColumnLengthMismatchError):
        Table({"a": [1, 2], "b": [3]})
