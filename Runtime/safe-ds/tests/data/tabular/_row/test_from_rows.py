import pytest
from safeds.data.tabular import Row, Table
from safeds.exceptions import MissingDataError


def test_from_rows() -> None:
    table_expected = Table.from_csv("tests/resources/test_row_table.csv")
    rows_is: list[Row] = table_expected.to_rows()
    table_is: Table = Table.from_rows(rows_is)

    assert table_is == table_expected


def test_from_rows_invalid() -> None:
    with pytest.raises(MissingDataError):
        Table.from_rows([])
