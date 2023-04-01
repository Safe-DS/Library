import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateColumnNameError, UnknownColumnNameError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("name_from", "name_to", "column_one", "column_two"),
    [("A", "D", "D", "B"), ("A", "A", "A", "B")],
)
def test_rename_valid(name_from: str, name_to: str, column_one: str, column_two: str) -> None:
    table: Table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    renamed_table = table.rename_column(name_from, name_to)
    assert renamed_table.schema.has_column(column_one)
    assert renamed_table.schema.has_column(column_two)
    assert renamed_table.count_columns() == 2


@pytest.mark.parametrize(
    ("name_from", "name_to", "error"),
    [
        ("C", "D", UnknownColumnNameError),
        ("A", "B", DuplicateColumnNameError),
        ("D", "D", UnknownColumnNameError),
    ],
)
def test_rename_invalid(name_from: str, name_to: str, error: Exception) -> None:
    table: Table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    with pytest.raises(error):
        table.rename_column(name_from, name_to)
