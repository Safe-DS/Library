import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.exceptions import MissingDataError
from tests.fixtures import resolve_resource_path


def test_from_rows() -> None:
    table_expected = Table.from_csv(resolve_resource_path("test_row_table.csv"))
    rows_is: list[Row] = table_expected.to_rows()
    table_is: Table = Table.from_rows(rows_is)

    assert table_is == table_expected


def test_from_rows_invalid() -> None:
    with pytest.raises(MissingDataError):
        Table.from_rows([])
