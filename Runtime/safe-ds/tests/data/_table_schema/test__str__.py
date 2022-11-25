import pandas as pd
from safe_ds.data import Table


def test__str__():
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": [1]}))
    assert (
        str(table.schema)
        == "TableSchema:\nColumn Count: 2\nColumns:\n    col1, object\n    col2, int64\n"
    )
