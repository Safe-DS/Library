from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Int
from tests.helpers import resolve_resource_path


def test_get_type_of_column() -> None:
    table = Table.from_json_file(resolve_resource_path("test_schema_table.json"))
    table_column_type = table.schema.get_type_of_column("A")
    assert table_column_type == Int()
