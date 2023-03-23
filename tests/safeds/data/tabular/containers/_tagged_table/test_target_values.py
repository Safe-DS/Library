from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table, TaggedTable


def test_tagged_table_target_values() -> None:
    table = Table.from_csv(resolve_resource_path("test_tagged_table.csv"))
    tagged_table = TaggedTable(table, "T")
    assert tagged_table.target_values._data[0] == 0
    assert tagged_table.target_values._data[1] == 1
