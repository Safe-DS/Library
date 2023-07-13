import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Integer, Schema
from safeds.exceptions import ColumnLengthMismatchError


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), Schema({})),
        (Table(), Schema({})),
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
    with pytest.raises(ColumnLengthMismatchError, match=r"The length of at least one column differs: \na: 2\nb: 1"):
        Table({"a": [1, 2], "b": [3]})
