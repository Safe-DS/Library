import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError
from tests.helpers import resolve_resource_path


def test_keep_columns() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    transformed_table = table.keep_only_columns(["A"])
    assert transformed_table.schema.has_column(
        "A"
    ) and not transformed_table.schema.has_column("B")


def test_keep_columns_warning() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    with pytest.raises(UnknownColumnNameError):
        table.keep_only_columns(["C"])
