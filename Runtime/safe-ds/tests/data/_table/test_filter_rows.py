import pandas as pd
import pytest
from safe_ds.data import Table


def test_filter_rows_valid():
    table = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    result_table = table.filter_rows(
        lambda row: row.get_value_by_column_name("col1") == 1
    )
    assert (
        result_table._data["col1"][0] == 1
        and result_table._data["col2"][1] == 4
        and result_table._data.size == 4
    )


def test_filter_rows_invalid():
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3], "col2": [1, 1, 4]}))
    with pytest.raises(TypeError):
        table.filter_rows(table._data["col1"] > table._data["col2"])
