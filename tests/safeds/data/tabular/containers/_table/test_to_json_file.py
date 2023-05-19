from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["by String", "empty"]
)
def test_should_create_json_file_from_table_by_str(table: Table) -> None:
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_json_file(tmp_file.name)
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_json_file(tmp_file.name)
    assert table == table_r


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["by String", "empty"]
)
def test_should_create_json_file_from_table_by_path(table: Table) -> None:
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_json_file(Path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_json_file(Path(tmp_file.name))
    assert table == table_r
