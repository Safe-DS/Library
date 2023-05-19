from pathlib import Path

import pytest
from safeds.data.tabular.containers import Table

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
        (
            resolve_resource_path("./empty_excel_file.xlsx"),
            Table()
        ),
    ],
    ids=["string path", "object path", "empty file"],
)
def test_should_create_table_from_excel_file(path: str | Path, expected: Table) -> None:
    table = Table.from_excel_file(path)
    assert table == expected


def test_should_raise_if_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_excel_file(resolve_resource_path("test_table_from_excel_file_invalid.xls"))
