from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import WrongFileExtensionError


def test_should_create_csv_file_from_table_by_str() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".xlsx") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_excel_file(tmp_file.name)
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_excel_file(tmp_file.name)
    assert table == table_r


def test_should_create_csv_file_from_table_by_path() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".xlsx") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_excel_file(Path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_excel_file(Path(tmp_file.name))
    assert table == table_r


def test_should_raise_error_if_wrong_file_extension() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".invalid_file_extension") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            with pytest.raises(WrongFileExtensionError):
                table.to_excel_file(Path(tmp_file.name))
