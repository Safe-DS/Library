import pytest
from safe_ds.data import Table


def test_read_json_valid() -> None:
    table = Table.from_json("tests/resources/test_table_read_json.json")
    assert table._data["A"][0] == 1 and table._data["B"][0] == 2


def test_read_json_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_json("tests/resources/test_table_read_json_invalid.json")
