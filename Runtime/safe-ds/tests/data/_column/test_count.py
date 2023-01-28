import pandas as pd
from safeds.data import Table


def test_count_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4, 5], "col2": [2, 3, 4, 5, 6]}))
    assert table.get_column("col1").count() == 5
