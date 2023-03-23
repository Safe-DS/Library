import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


def test_table_column_keep() -> None:
    table = Table.from_csv(resolve_resource_path("test_table_read_csv.csv"))
    transformed_table = table.keep_columns(["A"])
    assert transformed_table.schema.has_column(
        "A"
    ) and not transformed_table.schema.has_column("B")


def test_table_column_keep_warning() -> None:
    table = Table.from_csv(resolve_resource_path("test_table_read_csv.csv"))
    with pytest.raises(UnknownColumnNameError):
        table.keep_columns(["C"])
