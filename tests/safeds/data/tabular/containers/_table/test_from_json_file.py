from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import WrongFileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize("path", ["table.json", Path("table.json")], ids=["by string", "by path"])
def test_should_create_table_from_json_file(path: str | Path) -> None:
    table = Table.from_json_file(resolve_resource_path(path))
    assert table.get_column("A").get_value(0) == 1
    assert table.get_column("B").get_value(0) == 2


@pytest.mark.parametrize(
    "path",
    ["test_table_from_json_file_invalid.json", Path("test_table_from_json_file_invalid.json")],
    ids=["by string", "by path"],
)
def test_should_raise_error_if_file_not_found(path: str | Path) -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_json_file(resolve_resource_path(path))


@pytest.mark.parametrize(
    "path",
    ["invalid_file_extension.file_extension", Path("invalid_file_extension.file_extension")],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_wrong_file_extension(path: str | Path) -> None:
    with pytest.raises(WrongFileExtensionError):
        Table.from_json_file(resolve_resource_path(path))
