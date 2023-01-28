import pandas as pd
from safeds.data import Table


def test__str__() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": [1]}))
    assert (
        str(table.schema)
        == "TableSchema:\nColumn Count: 2\nColumns:\n    col1: string\n    col2: int\n"
    )
