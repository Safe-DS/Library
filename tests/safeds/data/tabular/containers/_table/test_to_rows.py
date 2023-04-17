from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import Integer, Schema, String

from tests.helpers import resolve_resource_path


def test_to_rows() -> None:
    table = Table.from_csv_file(resolve_resource_path("test_row_table.csv"))
    expected_schema: Schema = Schema(
        {
            "A": Integer(),
            "B": Integer(),
            "D": String(),
        },
    )
    rows_expected: list[Row] = [
        Row([1, 4, "d"], expected_schema),
        Row([2, 5, "e"], expected_schema),
        Row([3, 6, "f"], expected_schema),
    ]

    rows_is: list[Row] = table.to_rows()

    for row_is, row_expected in zip(rows_is, rows_expected, strict=True):
        assert row_is == row_expected
