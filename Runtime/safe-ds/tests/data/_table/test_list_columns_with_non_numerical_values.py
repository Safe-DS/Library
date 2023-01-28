import pandas as pd
from safeds.data import Table


def test_list_columns_with_non_numerical_values_valid() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C", "A"],
                "col2": ["Test1", "Test1", "Test3", "Test1"],
                "col3": [1, 2, 3, 4],
                "col4": [2, 3, 1, 4],
            }
        )
    )
    columns = table.list_columns_with_non_numerical_values()
    assert columns[0] == table.get_column("col1")
    assert columns[1] == table.get_column("col2")
    assert len(columns) == 2
