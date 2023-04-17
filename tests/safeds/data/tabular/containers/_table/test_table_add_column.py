import pytest

from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import ColumnSizeError, DuplicateColumnNameError
from tests.helpers import resolve_resource_path


def test_table_add_column_valid() -> None:
    input_table = Table.from_csv_file(resolve_resource_path("test_table_add_column_valid_input.csv"))
    expected = Table.from_csv_file(resolve_resource_path("test_table_add_column_valid_output.csv"))
    column = Column("C", ["a", "b", "c"])

    result = input_table.add_column(column)
    assert expected == result


@pytest.mark.parametrize(
    ("column_values", "column_name", "error"),
    [
        (["a", "b", "c"], "B", DuplicateColumnNameError),
        (["a", "b"], "C", ColumnSizeError),
    ],
)
def test_table_add_column_(column_values: list[str], column_name: str, error: type[Exception]) -> None:
    input_table = Table.from_csv_file(resolve_resource_path("test_table_add_column_valid_input.csv"))
    column = Column(column_name, column_values)

    with pytest.raises(error):
        input_table.add_column(column)
