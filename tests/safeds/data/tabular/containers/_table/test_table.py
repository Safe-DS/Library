from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import RealNumber, TableSchema


def test_create_empty_table() -> None:
    table = Table([], TableSchema({"col1": RealNumber()}))
    col = table.get_column("col1")
    assert col.count() == 0
    assert isinstance(col.type, RealNumber)
    assert col.name == "col1"


def test_create_empty_table_without_schema() -> None:
    table = Table([])
    assert table.schema == TableSchema({})
