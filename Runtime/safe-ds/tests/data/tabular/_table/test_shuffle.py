import pandas as pd
from safeds.data.tabular import Table


def test_shuffle_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [1]}))
    result_table = table.shuffle()
    assert table == result_table
