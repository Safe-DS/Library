import pandas as pd
from safe_ds.data import Table


def test_filter_rows_valid():
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    table2 = Table(pd.DataFrame(data={"col1": []}))
    assert table1.count_rows() == 3 and table2.count_rows() == 0
