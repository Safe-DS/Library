import pytest
from safeds.data.tabular import Table
from safeds.exceptions import IndexOutOfBoundsError


def test_get_row() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    val = table.get_row(0)
    assert val.get_value("A") == 1 and val.get_value("B") == 2


def test_get_row_negative_index() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(-1)


def test_get_row_out_of_bounds_index() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(5)
