from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import WrongFileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            resolve_resource_path("./dummy_excel_file.xlsx"),
            Table(
                {
                    "A": [1],
                    "B": [2],
                },
            ),
        ),
        (
            Path(resolve_resource_path("./dummy_excel_file.xlsx")),
            Table(
                {
                    "A": [1],
                    "B": [2],
                },
            ),
        ),
    ],
    ids=["string path", "object path"],
)
def test_should_create_table_from_excel_file(path: str | Path, expected: Table) -> None:
    table = Table.from_excel_file(path)
    assert table == expected


def test_should_raise_if_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match=r"test_table_from_excel_file_invalid.xls\" does not exist"):
        Table.from_excel_file(resolve_resource_path("test_table_from_excel_file_invalid.xls"))


@pytest.mark.parametrize(
    ("path", "expected_error_message"),
    [("invalid_file_extension.file_extension", r"invalid_file_extension.file_extension has a wrong file extension. Please provide a file with the following extension\(s\): \['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'\]"), (Path("invalid_file_extension.file_extension"), r"invalid_file_extension.file_extension has a wrong file extension. Please provide a file with the following extension\(s\): \['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'\]")],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_wrong_file_extension(path: str | Path, expected_error_message: str) -> None:
    with pytest.raises(WrongFileExtensionError, match=expected_error_message):
        Table.from_excel_file(resolve_resource_path(path))
