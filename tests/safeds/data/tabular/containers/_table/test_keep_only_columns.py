import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_keep_columns() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    transformed_table = table.keep_only_columns(["A"])
    assert transformed_table.schema.has_column("A")
    assert not transformed_table.schema.has_column("B")


def test_keep_columns_order() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    transformed_table = table.keep_only_columns(["B", "A"])
    assert table.to_columns()[0] == transformed_table.to_columns()[1]
    assert table.to_columns()[1] == transformed_table.to_columns()[0]
    assert table.get_column("A") == transformed_table.get_column("A")
    assert table.get_column("B") == transformed_table.get_column("B")


def test_keep_columns_warning() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    with pytest.raises(UnknownColumnNameError):
        table.keep_only_columns(["C"])
