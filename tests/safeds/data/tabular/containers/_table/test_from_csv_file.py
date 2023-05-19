from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("table.csv", Table({"A": [1], "B": [2]})),
        (Path("table.csv"), Table({"A": [1], "B": [2]})),
        ("emptytable.csv", Table()),
    ],
    ids=["by String", "by path", "empty"],
)
def test_should_create_table_from_csv_file(path: str | Path, expected: Table) -> None:
    table = Table.from_csv_file(resolve_resource_path(path))
    assert table == expected


@pytest.mark.parametrize(
    "path",
    ["test_table_from_csv_file_invalid.csv", Path("test_table_from_csv_file_invalid.csv")],
    ids=["by String", "by path"],
)
def test_should_raise_error_if_file_not_found(path: str | Path) -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_csv_file(resolve_resource_path(path))
