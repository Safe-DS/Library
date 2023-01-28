import pandas as pd
from safeds.data import IntColumnType, Row, StringColumnType, Table, TableSchema


def test_to_rows() -> None:
    table = Table.from_csv("tests/resources/test_row_table.csv")
    expected_schema: TableSchema = TableSchema(
        {
            "A": IntColumnType(),
            "B": IntColumnType(),
            "D": StringColumnType(),
        }
    )
    rows_expected: list[Row] = [
        Row(pd.Series([1, 4, "d"], index=["A", "B", "D"], name=0), expected_schema),
        Row(pd.Series([2, 5, "e"], index=["A", "B", "D"], name=0), expected_schema),
        Row(pd.Series([3, 6, "f"], index=["A", "B", "D"], name=0), expected_schema),
    ]

    rows_is: list[Row] = table.to_rows()

    for row_is, row_expected in zip(rows_is, rows_expected):
        assert row_is == row_expected
