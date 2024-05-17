from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import FileExtensionError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["by String", "empty"],
)
def test_should_create_csv_file_from_table_by_str(table: Table) -> None:
    with NamedTemporaryFile(suffix=".csv") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_csv_file(resolve_resource_path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_csv_file(tmp_file.name)

    # We test all since polars might raise if files are empty
    # TODO: extract to tests for schema, number of columns etc.
    assert table.schema == table_r.schema
    assert table.column_count == table_r.column_count
    assert table_r == table


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["by String", "empty"],
)
def test_should_create_csv_file_from_table_by_path(table: Table) -> None:
    with NamedTemporaryFile(suffix=".csv") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_csv_file(Path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_csv_file(Path(tmp_file.name))
    assert table.schema == table_r.schema
    assert table.column_count == table_r.column_count
    assert table == table_r


def test_should_raise_error_if_wrong_file_extension() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile(suffix=".invalid_file_extension") as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file, pytest.raises(FileExtensionError):
            table.to_csv_file(Path(tmp_file.name))
