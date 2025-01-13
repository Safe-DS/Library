import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "name", "expected"),
    [
        (
            Table({"col1": [1]}),
            "col1",
            ColumnType.int64(),
        ),
        (
            Table({"col1": ["a"]}),
            "col1",
            ColumnType.string(),
        ),
    ],
    ids=["int column", "string column"],
)
def test_should_return_type_of_column(table: Table, name: str, expected: ColumnType) -> None:
    row = _LazyVectorizedRow(table)
    assert row.get_column_type(name) == expected


def test_should_raise_if_column_name_is_unknown() -> None:
    row = _LazyVectorizedRow(Table({}))
    with pytest.raises(ColumnNotFoundError):
        row.get_column_type("col1")
