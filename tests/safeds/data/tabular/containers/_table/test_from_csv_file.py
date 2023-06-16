from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import WrongFileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize("path", ["table.csv", Path("table.csv")], ids=["by String", "by path"])
def test_should_create_table_from_csv_file(path: str | Path) -> None:
    table1 = Table.from_csv_file(resolve_resource_path(path))
    table2 = Table({"A": [1], "B": [2]})
    assert table1.schema == table2.schema
    assert table1 == table2


@pytest.mark.parametrize(
    ("path", "expected_error_message"),
    [
        ("test_table_from_csv_file_invalid.csv", r"test_table_from_csv_file_invalid.csv\" does not exist"),
        (Path("test_table_from_csv_file_invalid.csv"), r"test_table_from_csv_file_invalid.csv\" does not exist"),
    ],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_file_not_found(path: str | Path, expected_error_message: str) -> None:
    with pytest.raises(FileNotFoundError, match=expected_error_message):
        Table.from_csv_file(resolve_resource_path(path))


@pytest.mark.parametrize(
    ("path", "expected_error_message"),
    [
        (
            "invalid_file_extension.file_extension",
            (
                r"invalid_file_extension.file_extension has a wrong file extension. Please provide a file with the"
                r" following extension\(s\): .csv"
            ),
        ),
        (
            Path("invalid_file_extension.file_extension"),
            (
                r"invalid_file_extension.file_extension has a wrong file extension. Please provide a file with the"
                r" following extension\(s\): .csv"
            ),
        ),
    ],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_wrong_file_extension(path: str | Path, expected_error_message: str) -> None:
    with pytest.raises(WrongFileExtensionError, match=expected_error_message):
        Table.from_csv_file(resolve_resource_path(path))
