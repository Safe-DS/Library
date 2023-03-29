import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError
from tests.helpers import resolve_resource_path


def test_transform_column_valid() -> None:
    input_table: Table = Table.from_csv_file(resolve_resource_path("test_table_transform_column.csv"))

    result: Table = input_table.transform_column("A", lambda row: row.get_value("A") * 2)

    assert result == Table.from_csv_file(resolve_resource_path("test_table_transform_column_output.csv"))


def test_transform_column_invalid() -> None:
    input_table: Table = Table.from_csv_file(resolve_resource_path("test_table_transform_column.csv"))

    with pytest.raises(UnknownColumnNameError):
        input_table.transform_column("D", lambda row: row.get_value("A") * 2)
