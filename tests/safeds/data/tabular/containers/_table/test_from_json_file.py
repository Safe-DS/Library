from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import WrongFileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("table.json", Table({"A": ["❔"], "B": [2]})),
        (Path("table.json"), Table({"A": ["❔"], "B": [2]})),
        (Path("emptytable.json"), Table()),
    ],
    ids=["by string", "by path", "empty"],
)
def test_should_create_table_from_json_file(path: str | Path, expected: Table) -> None:
    table = Table.from_json_file(resolve_resource_path(path))
    assert table.schema == expected.schema
    assert table == expected


@pytest.mark.parametrize(
    ("path", "expected_error_message"),
    [
        ("test_table_from_json_file_invalid.json", r"test_table_from_json_file_invalid.json\" does not exist"),
        (Path("test_table_from_json_file_invalid.json"), r"test_table_from_json_file_invalid.json\" does not exist"),
    ],
    ids=["by string", "by path"],
)
def test_should_raise_error_if_file_not_found(path: str | Path, expected_error_message: str) -> None:
    with pytest.raises(FileNotFoundError, match=expected_error_message):
        Table.from_json_file(resolve_resource_path(path))


@pytest.mark.parametrize(
    ("path", "expected_error_message"),
    [
        (
            "invalid_file_extension.file_extension",
            (
                r"invalid_file_extension.file_extension has a wrong file extension. Please provide a file with the"
                r" following extension\(s\): .json"
            ),
        ),
        (
            Path("invalid_file_extension.file_extension"),
            (
                r"invalid_file_extension.file_extension has a wrong file extension. Please provide a file with the"
                r" following extension\(s\): .json"
            ),
        ),
    ],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_wrong_file_extension(path: str | Path, expected_error_message: str) -> None:
    with pytest.raises(WrongFileExtensionError, match=expected_error_message):
        Table.from_json_file(resolve_resource_path(path))
