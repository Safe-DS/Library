import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import MissingDataError
from safeds.data.tabular.typing import Integer, Schema, String


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table([[1, 4, "d"], [2, 5, "e"], [3, 6, "f"]], Schema({"A": Integer(), "B": Integer(), "D": String()})),
            Table([[1, 4, "d"], [2, 5, "e"], [3, 6, "f"]], Schema({"A": Integer(), "B": Integer(), "D": String()})),
        ),
    ],
    ids=["empty"],
)
def test_should_create_table_from_rows(table: Table, expected: Table) -> None:
    rows_is = table.to_rows()
    table_is = Table.from_rows(rows_is)

    assert table_is == expected


def test_should_raise_error_if_data_missing() -> None:
    with pytest.raises(MissingDataError):
        Table.from_rows([])
