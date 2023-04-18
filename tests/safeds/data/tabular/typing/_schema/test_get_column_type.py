from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Integer


def test_get_type_of_column() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    table_column_type = table.schema.get_type_of_column("A")
    assert table_column_type == Integer()
