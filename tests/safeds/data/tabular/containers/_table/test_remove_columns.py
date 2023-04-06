import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_table_remove_columns() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    transformed_table = table.remove_columns(["A"])
    assert transformed_table.schema.has_column("B")
    assert not transformed_table.schema.has_column("A")


def test_table_remove_columns_warning() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    with pytest.raises(UnknownColumnNameError):
        table.remove_columns(["C"])
