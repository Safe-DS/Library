import pytest

from safeds.data.tabular.typing import ColumnType, Schema
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("schema", "name", "expected"),
    [
        (
            Schema({"col1": ColumnType.int64()}),
            "col1",
            ColumnType.int64(),
        ),
        (
            Schema({"col1": ColumnType.string()}),
            "col1",
            ColumnType.string(),
        ),
    ],
    ids=["int column", "string column"],
)
def test_should_return_data_type_of_column(schema: Schema, name: str, expected: ColumnType) -> None:
    assert schema.get_column_type(name) == expected


def test_should_raise_if_column_name_is_unknown() -> None:
    with pytest.raises(ColumnNotFoundError):
        Schema({}).get_column_type("col1")
