from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import WrongFileExtensionError


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["by String", "empty"],
)
def test_should_create_excel_file_from_table_by_str(table: Table) -> None:
    with NamedTemporaryFile(suffix=".xlsx") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_excel_file(tmp_file.name)
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_excel_file(tmp_file.name)
    assert table.schema == table_r.schema
    assert table == table_r


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["by String", "empty"],
)
def test_should_create_excel_file_from_table_by_path(table: Table) -> None:
    with NamedTemporaryFile(suffix=".xlsx") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_excel_file(Path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_excel_file(Path(tmp_file.name))
    assert table.schema == table_r.schema
    assert table == table_r


def test_should_raise_error_if_wrong_file_extension() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".invalid_file_extension") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file, pytest.raises(
            WrongFileExtensionError,
            match=(
                r".invalid_file_extension has a wrong file extension. Please provide a file with the following"
                r" extension\(s\): \['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'\]"
            ),
        ):
            table.to_excel_file(Path(tmp_file.name))
