import pytest
from safeds.data.tabular import Table


def test_read_json_valid() -> None:
    table = Table.from_json("tests/resources/test_table_read_json.json")
    assert (
        table.get_column("A").get_value(0) == 1
        and table.get_column("B").get_value(0) == 2
    )


def test_read_json_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_json("tests/resources/test_table_read_json_invalid.json")
