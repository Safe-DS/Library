from safeds.data.tabular.containers import Table, TaggedTable
from tests.fixtures import resolve_resource_path


def test_tagged_table_target() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_tagged_table.csv"))
    tagged_table = TaggedTable(table, "T")
    assert tagged_table.target._data[0] == 0
    assert tagged_table.target._data[1] == 1
