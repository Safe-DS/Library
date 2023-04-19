import numpy as np
from safeds.data.tabular.containers import Table


def test_count_null_values_valid() -> None:
    table = Table.from_dict({"col1": [1, 2, 3, 4, 5], "col2": [None, None, 1, np.nan, np.nan]})
    empty_table = Table.from_dict({"col1": []})
    column1 = table.get_column("col1")
    column2 = table.get_column("col2")
    empty_column = empty_table.get_column("col1")
    assert column1._count_missing_values() == 0
    assert column2._count_missing_values() == 4
    assert empty_column._count_missing_values() == 0
