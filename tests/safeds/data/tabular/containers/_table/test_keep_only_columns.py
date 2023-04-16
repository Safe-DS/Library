import pytest
from safeds.data.tabular.containers import Table, Column
from safeds.data.tabular.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_keep_only_columns() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    transformed_table = table.keep_only_columns(["A"])
    assert transformed_table.schema.has_column("A")
    assert not transformed_table.schema.has_column("B")


def test_keep_only_columns_order() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    transformed_table = table.keep_only_columns(["B", "A"])
    assert transformed_table == Table.from_columns(
        [
            Column("B", [2]),
            Column("A", [1])
        ]
    )


def test_keep_only_columns_warning() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_from_csv_file.csv"))
    with pytest.raises(UnknownColumnNameError):
        table.keep_only_columns(["C"])
