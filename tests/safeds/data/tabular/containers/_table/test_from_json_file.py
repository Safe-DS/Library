from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import FileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("json/empty.json", Table({})),
        ("json/non-empty.json", Table({"A": [1], "B": [2]})),
        ("json/special-character.json", Table({"A": ["❔"], "B": [2]})),
    ],
    ids=["empty", "non-empty", "special character"],
)
class TestShouldCreateTableFromJsonFile:
    def test_path_as_string(self, path: str, expected: Table) -> None:
        path_as_string = resolve_resource_path(path)
        actual = Table.from_json_file(path_as_string)
        assert actual == expected

    def test_path_as_path_object(self, path: str, expected: Table) -> None:
        path_as_path_object = Path(resolve_resource_path(path))
        actual = Table.from_json_file(path_as_path_object)
        assert actual == expected


def test_should_raise_error_if_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_json_file(resolve_resource_path("not-found.json"))


def test_should_raise_error_if_wrong_file_extension() -> None:
    with pytest.raises(FileExtensionError):
        Table.from_json_file(resolve_resource_path("invalid-extension.txt"))
