from pathlib import Path

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import FileExtensionError
from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("parquet/empty.parquet", Table({})),
        ("parquet/non-empty.parquet", Table({"A": [1], "B": [2]})),
        ("parquet/special-character.parquet", Table({"A": ["❔"], "B": [2]})),
        ("parquet/empty", Table({})),
    ],
    ids=[
        "empty",
        "non-empty",
        "special character",
        "missing extension",
    ],
)
class TestShouldCreateTableFromParquetFile:
    def test_path_as_string(self, path: str, expected: Table) -> None:
        path_as_string = resolve_resource_path(path)
        actual = Table.from_parquet_file(path_as_string)
        assert actual == expected
        expected.to_parquet_file(path_as_string)

    def test_path_as_path_object(self, path: str, expected: Table) -> None:
        path_as_path_object = Path(resolve_resource_path(path))
        actual = Table.from_parquet_file(path_as_path_object)
        assert actual == expected


def test_should_raise_if_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_parquet_file(resolve_resource_path("not-found.parquet"))


def test_should_raise_if_wrong_file_extension() -> None:
    with pytest.raises(FileExtensionError):
        Table.from_parquet_file(resolve_resource_path("invalid-extension.txt"))
