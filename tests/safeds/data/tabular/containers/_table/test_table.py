import numpy as np
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType, TableSchema


def test_create_empty_table() -> None:
    table = Table([], TableSchema({"col1": ColumnType.from_numpy_dtype(np.dtype(float))}))
    col = table.get_column("col1")
    assert col.count() == 0
    assert isinstance(col.type, type(ColumnType.from_numpy_dtype(np.dtype(float))))
    assert col.name == "col1"


def test_create_empty_table_without_schema() -> None:
    table = Table([])
    assert table.schema == TableSchema({})
