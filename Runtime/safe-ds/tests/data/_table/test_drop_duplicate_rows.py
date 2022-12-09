import pandas as pd
import pytest
from safe_ds.data import Table


@pytest.mark.parametrize(
    "path",
    [
        "tests/resources/test_table_duplicate_rows_duplicates.csv",
        "tests/resources/test_table_duplicate_rows_no_duplicates.csv",
    ],
)
def test_drop_duplicate_rows(path: str) -> None:
    expected_table = Table(pd.DataFrame(data={"A": [1, 4], "B": [2, 5]}))
    table = Table.from_csv(path)
    result_table = table.drop_duplicate_rows()
    assert expected_table == result_table
