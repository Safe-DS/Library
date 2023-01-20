import pandas as pd
from safe_ds.data import Table


def test_shuffle_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [1]}))
    result_table = table.shuffle()
    assert table == result_table
