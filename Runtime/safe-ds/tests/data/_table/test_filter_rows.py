import pandas as pd
import pytest
from safeds.data import Table


def test_filter_rows_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    result_table = table.filter_rows(lambda row: row.get_value("col1") == 1)
    assert (
        result_table.get_column("col1").get_value(0) == 1
        and result_table.get_column("col2").get_value(1) == 4
        and result_table.count_columns() == 2
        and result_table.count_rows() == 2
    )


# noinspection PyTypeChecker
def test_filter_rows_invalid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3], "col2": [1, 1, 4]}))
    with pytest.raises(TypeError):
        table.filter_rows(
            table.get_column("col1")._data > table.get_column("col2")._data
        )
