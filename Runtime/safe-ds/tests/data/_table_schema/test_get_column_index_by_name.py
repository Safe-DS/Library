import pandas as pd
from safe_ds.data import Table


def test_get_column_index_by_name():
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
    assert (
        table.schema._get_column_index_by_name("col1") == 0
        and table.schema._get_column_index_by_name("col2") == 1
    )
