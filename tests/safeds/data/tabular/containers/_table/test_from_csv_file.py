from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import FileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("csv/empty.csv", Table({})),
        ("csv/non-empty.csv", Table({"A": [1], "B": [2]})),
        ("csv/special-character.csv", Table({"A": ["❔"], "B": [2]})),
    ],
    ids=["empty", "non-empty", "special character"],
)
class TestShouldCreateTableFromCsvFile:
    def test_path_as_string(self, path: str, expected: Table) -> None:
        path_as_string = resolve_resource_path(path)
        actual = Table.from_csv_file(path_as_string)
        assert actual.schema == expected.schema
        assert actual == expected

    def test_path_as_path_object(self, path: str, expected: Table) -> None:
        path_as_path_object = Path(resolve_resource_path(path))
        actual = Table.from_csv_file(path_as_path_object)
        assert actual.schema == expected.schema
        assert actual == expected


def test_should_raise_error_if_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_csv_file(resolve_resource_path("not-found.csv"))


def test_should_raise_error_if_wrong_file_extension() -> None:
    with pytest.raises(FileExtensionError):
        Table.from_csv_file(resolve_resource_path("invalid-extension.txt"))
