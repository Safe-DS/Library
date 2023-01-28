import pytest
from safeds.data import Table


def test_read_csv_valid() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    assert (
        table.get_column("A").get_value(0) == 1
        and table.get_column("B").get_value(0) == 2
    )


def test_read_csv_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_csv("tests/resources/test_table_read_csv_invalid.csv")
