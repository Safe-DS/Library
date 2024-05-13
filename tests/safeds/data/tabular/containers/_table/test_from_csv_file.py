from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import FileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("table.csv", Table({"A": ["❔"], "B": [2]})),
        (Path("table.csv"), Table({"A": ["❔"], "B": [2]})),
        ("emptytable.csv", Table()),
    ],
    ids=["by String", "by path", "empty"],
)
def test_should_create_table_from_csv_file(path: str | Path, expected: Table) -> None:
    table = Table.from_csv_file(resolve_resource_path(path))
    assert table.schema == expected.schema
    assert table == expected


@pytest.mark.parametrize(
    "path",
    [
        "test_table_from_csv_file_invalid.csv",
        Path("test_table_from_csv_file_invalid.csv"),
    ],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_file_not_found(path: str | Path) -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_csv_file(resolve_resource_path(path))


@pytest.mark.parametrize(
    "path",
    [
        "invalid_file_extension.file_extension",
        Path("invalid_file_extension.file_extension"),
    ],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_wrong_file_extension(path: str | Path) -> None:
    with pytest.raises(FileExtensionError):
        Table.from_csv_file(resolve_resource_path(path))
