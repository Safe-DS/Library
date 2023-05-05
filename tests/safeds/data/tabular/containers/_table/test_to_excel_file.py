from pathlib import Path
from tempfile import NamedTemporaryFile

from safeds.data.tabular.containers import Table


def test_should_create_csv_file_from_table_by_str() -> None:
    table = Table.from_dict({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".xlsx") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_excel_file(tmp_file.name)
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_excel_file(tmp_file.name)
    assert table == table_r


def test_should_create_csv_file_from_table_by_path() -> None:
    table = Table.from_dict({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".xlsx") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_excel_file(Path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_excel_file(Path(tmp_file.name))
    assert table == table_r
