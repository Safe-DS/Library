from safeds.data.tabular.containers import Table, TaggedTable
from tests.fixtures import resolve_resource_path


def test_tagged_table_features() -> None:
    table = Table.from_csv(resolve_resource_path("test_tagged_table.csv"))
    tagged_table = TaggedTable(table, "T")
    assert "T" not in tagged_table.features._data
    assert tagged_table.features.schema.has_column("A")
    assert tagged_table.features.schema.has_column("B")
    assert tagged_table.features.schema.has_column("C")
