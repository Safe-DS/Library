import pytest

from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.typing import Integer, Schema


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (
            [
                Column("A", [1, 4]),
                Column("B", [2, 5]),
            ],
            Table([[1, 2], [4, 5]], Schema({"A": Integer(), "B": Integer()}))
        ),
    ],
)
def test_from_columns(columns: list[Column], expected: Table) -> None:
    assert Table.from_columns(columns) == expected
