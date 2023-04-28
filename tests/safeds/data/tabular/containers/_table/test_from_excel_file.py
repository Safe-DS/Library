from pathlib import Path

import openpyxl
import pytest
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("path"),
    [
        "./dummy_excel_file.xlsx",
        Path("./dummy_excel_file.xlsx")
    ],
    ids=["string path", "object path"],
)
def test_from_excel_file_valid(path) -> None:
    table = Table.from_excel_file(path)
    assert table.get_column("A").get_value(0) == 1
    assert table.get_column("B").get_value(0) == 2


def test_from_excel_file_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_excel_file(resolve_resource_path("test_table_from_excel_file_invalid.xls"))
