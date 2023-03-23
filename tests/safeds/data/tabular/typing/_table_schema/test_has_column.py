from safeds.data.tabular.containers import Table
from tests.fixtures import resolve_resource_path


def test_has_column_true() -> None:
    table = Table.from_json(resolve_resource_path("test_schema_table.json"))

    assert table.schema.has_column("A")


def test_has_column_false() -> None:
    table = Table.from_json(resolve_resource_path("test_schema_table.json"))

    assert not table.schema.has_column("C")
