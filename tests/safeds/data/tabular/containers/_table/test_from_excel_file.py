
from pathlib import Path

import openpyxl
import pytest
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_from_excel_file_valid() -> None:
    path = "./test.xlsx"
    table = Table.from_dict({"A": ["1"], "B": ["2"]})
    tmp_table_file = openpyxl.Workbook()
    tmp_table_file.save(path)
    try:
        with Path(path).open("w", encoding="utf-8") as _:
            table.to_excel_file(path)
        table = Table.from_excel_file(path)
        assert table.get_column("A").get_value(0) == 1
        assert table.get_column("B").get_value(0) == 2
    finally:
        Path(path).unlink()


def test_from_excel_file_invalid() -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_excel_file(resolve_resource_path("test_table_from_excel_file_invalid.xls"))
