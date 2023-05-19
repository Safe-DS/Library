from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("table.json", Table({"A": [1], "B": [2]})),
        (Path("table.json"), Table({"A": [1], "B": [2]})),
        (Path("emptytable.json"), Table())
    ],
    ids=["by string", "by path", "empty"])
def test_should_create_table_from_json_file(path: str | Path, expected: Table) -> None:
    table = Table.from_json_file(resolve_resource_path(path))
    assert table == expected


@pytest.mark.parametrize(
    "path",
    ["test_table_from_json_file_invalid.json", Path("test_table_from_json_file_invalid.json")],
    ids=["by string", "by path"],
)
def test_should_raise_error_if_file_not_found(path: str | Path) -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_json_file(path)
