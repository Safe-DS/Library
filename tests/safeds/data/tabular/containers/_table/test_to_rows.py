from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import Integer, Schema, String
import polars as pl

def test_to_rows() -> None:
    table = Table.from_dict(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "D": ["d", "e", "f"],
        },
    )

    expected_schema = Schema(
        {
            "A": Integer(),
            "B": Integer(),
            "D": String(),
        },
    )
    rows_expected = [
        Row(pl.DataFrame({"A": 1, "B": 4, "D": "d"}), expected_schema),
        Row(pl.DataFrame({"A": 2, "B": 5, "D": "e"}), expected_schema),
        Row(pl.DataFrame({"A": 3, "B": 6, "D": "f"}), expected_schema),
    ]

    rows_is = table.to_rows()

    for row_is, row_expected in zip(rows_is, rows_expected, strict=True):
        assert row_is == row_expected
