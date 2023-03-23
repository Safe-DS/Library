from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table, TaggedTable


def test_tagged_table_feature_vectors() -> None:
    table = Table.from_csv(resolve_resource_path("test_tagged_table.csv"))
    tagged_table = TaggedTable(table, "T")
    assert "T" not in tagged_table.feature_vectors._data
    assert tagged_table.feature_vectors.schema.has_column("A")
    assert tagged_table.feature_vectors.schema.has_column("B")
    assert tagged_table.feature_vectors.schema.has_column("C")
