from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import RealNumber, Schema


def test_remove_rows_with_missing_values_valid() -> None:
    table = Table.from_dict(
        {
            "col1": [None, None, "C", "A"],
            "col2": [None, "Test1", "Test3", "Test1"],
            "col3": [None, 2, 3, 4],
            "col4": [None, 3, 1, 4],
        },
    )
    updated_table = table.remove_rows_with_missing_values()
    assert updated_table.number_of_rows == 2


def test_remove_rows_with_missing_values_empty() -> None:
    table = Table([], Schema({"col1": RealNumber()}))
    updated_table = table.remove_rows_with_missing_values()
    assert updated_table.column_names == ["col1"]
