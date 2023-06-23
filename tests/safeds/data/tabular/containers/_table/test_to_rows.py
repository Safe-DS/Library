import pandas as pd
import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import Integer, Schema, String


@pytest.mark.parametrize(
    ("table", "rows_expected"),
    [
        (
            Table._from_pandas_dataframe(
                pd.DataFrame([[1, 4, "d"], [2, 5, "e"], [3, 6, "f"]]),
                Schema({"A": Integer(), "B": Integer(), "D": String()}),
            ),
            [
                Row._from_pandas_dataframe(
                    pd.DataFrame({"A": [1], "B": [4], "D": ["d"]}),
                    Schema({"A": Integer(), "B": Integer(), "D": String()}),
                ),
                Row._from_pandas_dataframe(
                    pd.DataFrame({"A": [2], "B": [5], "D": ["e"]}),
                    Schema({"A": Integer(), "B": Integer(), "D": String()}),
                ),
                Row._from_pandas_dataframe(
                    pd.DataFrame({"A": [3], "B": [6], "D": ["f"]}),
                    Schema({"A": Integer(), "B": Integer(), "D": String()}),
                ),
            ],
        ),
        (Table(), []),
    ],
    ids=["normal", "empty"],
)
def test_should_return_list_of_rows(table: Table, rows_expected: list[Row]) -> None:
    rows_is = table.to_rows()

    for row_is, row_expected in zip(rows_is, rows_expected, strict=True):
        assert row_is.schema == row_expected.schema
        assert row_is == row_expected
