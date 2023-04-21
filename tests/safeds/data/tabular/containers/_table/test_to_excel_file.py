from pathlib import Path

import openpyxl
from safeds.data.tabular.containers import Table


def test_to_excel_file() -> None:
    table = Table.from_dict({"col1": ["col1_1"], "col2": ["col2_1"]})
    path = "./test.xlsx"

    tmp_table_file = openpyxl.Workbook()
    tmp_table_file.save(path)
    try:
        with Path(path).open("w", encoding="utf-8") as _:
            table.to_excel_file(path)
        with Path(path).open("r", encoding="utf-8") as _:
            table_r = Table.from_excel_file(path)
        assert table == table_r
    finally:
        Path(path).unlink()
