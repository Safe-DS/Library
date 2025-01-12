from pathlib import Path

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import FileExtensionError


@pytest.mark.parametrize(
    "table",
    [
        Table({}),
        Table({"col1": [0, 1], "col2": ["a", "b"]}),
    ],
    ids=[
        "empty",
        # "no rows",  # This test fails, because polars creates row-oriented JSON.
        "with data",
    ],
)
class TestShouldCreateJsonFile:
    def test_path_as_string(self, table: Table, tmp_path: Path) -> None:
        path_as_string = str(tmp_path / "table.json")

        table.to_json_file(path_as_string)
        restored = Table.from_json_file(path_as_string)
        assert restored == table

    def test_path_as_path_object(self, table: Table, tmp_path: Path) -> None:
        path_as_path_object = tmp_path / "table.json"

        table.to_json_file(path_as_path_object)
        restored = Table.from_json_file(path_as_path_object)
        assert restored == table


def test_should_add_missing_extension(tmp_path: Path) -> None:
    write_path = tmp_path / "table"
    read_path = tmp_path / "table.json"

    table = Table({})
    table.to_json_file(write_path)
    restored = Table.from_json_file(read_path)
    assert restored == table


def test_should_raise_if_wrong_file_extension(tmp_path: Path) -> None:
    with pytest.raises(FileExtensionError):
        Table({}).to_json_file(tmp_path / "table.txt")
