from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import Integer, Schema, String


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
        Row([1, 4, "d"], expected_schema),
        Row([2, 5, "e"], expected_schema),
        Row([3, 6, "f"], expected_schema),
    ]

    rows_is = table.to_rows()

    for row_is, row_expected in zip(rows_is, rows_expected, strict=True):
        assert row_is == row_expected
