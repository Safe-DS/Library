from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import RealNumber, Schema


def test_should_create_empty_table() -> None:
    table = Table([], Schema({"col1": RealNumber()}))
    col = table.get_column("col1")
    assert col.number_of_rows == 0
    assert isinstance(col.type, RealNumber)
    assert col.name == "col1"


def test_should_create_empty_table_without_schema() -> None:
    table = Table([])
    assert table.schema == Schema({})
