import numpy as np
import pandas as pd
from safeds.data.tabular import Table


def test_list_columns_with_missing_values() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1", 2], "col2": [np.nan, 3]}))
    columns_with_missing_values = table.list_columns_with_missing_values()
    assert columns_with_missing_values == [table.get_column("col2")]
