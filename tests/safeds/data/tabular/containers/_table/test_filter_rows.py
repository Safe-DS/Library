import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType, Integer, Schema, String


@pytest.mark.parametrize(
    ("table1", "filter_column", "filter_value", "table2"),
    [
        (
            Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            "col1",
            1,
            Table.from_dict({"col1": [1, 1], "col2": [1, 4]}),
        ),
        (
            Table.from_dict({"col1": [3, 2, 4], "col2": [1, 2, 4]}),
            "col1",
            1,
            Table([], Schema({"col1": Integer(), "col2": Integer()})),
        ),
    ],
    ids=["filter for col1 = 1", "empty table"],
)
def test_should_filter_rows(table1: Table, filter_column: String, filter_value: ColumnType, table2: Table) -> None:
    table1 = table1.filter_rows(lambda row: row.get_value(filter_column) == filter_value)
    assert table2 == table1


# noinspection PyTypeChecker
def test_should_raise_error_if_column_type_invalid() -> None:
    table = Table.from_dict({"col1": [1, 2, 3], "col2": [1, 1, 4]})
    with pytest.raises(TypeError):
        table.filter_rows(table.get_column("col1")._data > table.get_column("col2")._data)
