import numpy as np

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType


def test_get_type_of_column() -> None:
    table = Table.from_json(resolve_resource_path("test_schema_table.json"))
    table_column_type = table.schema.get_type_of_column("A")
    assert table_column_type == ColumnType.from_numpy_dtype(np.dtype("int64"))
