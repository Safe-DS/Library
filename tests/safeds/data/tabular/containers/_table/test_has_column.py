from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_has_column_positive() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_has_column.csv"))
    assert table.has_column("A")


def test_has_column_negative() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_table_has_column.csv"))
    assert not table.has_column("C")
