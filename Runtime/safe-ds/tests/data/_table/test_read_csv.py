import pytest
from safe_ds.data import Table


def test_read_csv_valid() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    assert table._data["A"][0] == 1 and table._data["B"][0] == 2


def test_read_csv_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_csv("tests/resources/test_table_read_csv_invalid.csv")
