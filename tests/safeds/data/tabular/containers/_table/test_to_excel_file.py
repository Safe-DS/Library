from pathlib import Path

import openpyxl
import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("path"),
    [
        "./temp_excel_file.xlsx",
        Path("./temp_excel_file.xlsx")
    ],
    ids=["string path", "object path"],
)
def test_to_excel_file(path) -> None:
    table = Table.from_dict({"col1": ["col1_1"], "col2": ["col2_1"]})
    try:
        with Path(path).open("w", encoding="utf-8") as _:
            table.to_excel_file(path)
        with Path(path).open("r", encoding="utf-8") as _:
            table_r = Table.from_excel_file(path)
        assert table == table_r
    finally:
        Path(path).unlink()
