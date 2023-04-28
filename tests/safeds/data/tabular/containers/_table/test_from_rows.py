import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import MissingDataError
from safeds.data.tabular.typing import Schema, Integer, String


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table([[1, 4, "d"], [2, 5, "e"], [3, 6, "f"]],
               Schema({"A": Integer(), "B": Integer(), "D": String()})),
         Table([[1, 4, "d"], [2, 5, "e"], [3, 6, "f"]],
               Schema({"A": Integer(), "B": Integer(), "D": String()}))),
    ],
    ids=["empty"]
)
def test_from_rows(table: Table, expected: Table) -> None:
    rows_is = expected.to_rows()
    table_is = Table.from_rows(rows_is)

    assert table_is == expected


def test_from_rows_invalid() -> None:
    with pytest.raises(MissingDataError):
        Table.from_rows([])
