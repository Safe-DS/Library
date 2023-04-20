from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import RealNumber, Schema


def test_remove_columns_with_non_numerical_values_valid() -> None:
    table = Table.from_dict(
        {
            "col1": ["A", "B", "C", "A"],
            "col2": ["Test1", "Test1", "Test3", "Test1"],
            "col3": [1, 2, 3, 4],
            "col4": [2, 3, 1, 4],
        },
    )
    updated_table = table.remove_columns_with_non_numerical_values()
    assert updated_table.column_names == ["col3", "col4"]


def test_remove_columns_with_non_numerical_values_empty() -> None:
    table = Table([], Schema({"col1": RealNumber()}))
    updated_table = table.remove_columns_with_non_numerical_values()
    assert updated_table.column_names == ["col1"]
