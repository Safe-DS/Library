import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


def test_table_remove_columns() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    transformed_table = table.remove_columns(["A"])
    assert transformed_table.schema.has_column("B")
    assert not transformed_table.schema.has_column("A")


def test_table_remove_columns_warning() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(UnknownColumnNameError):
        table.remove_columns(["C"])
