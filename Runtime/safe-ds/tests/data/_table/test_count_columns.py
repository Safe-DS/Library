import pandas as pd
from safeds.data import Table


def test_count_columns() -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    table2 = Table(pd.DataFrame(data={"col1": []}))
    assert table1.count_columns() == 2 and table2.count_columns() == 1
