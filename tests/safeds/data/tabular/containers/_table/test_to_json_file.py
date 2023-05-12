from pathlib import Path
from tempfile import NamedTemporaryFile

from safeds.data.tabular.containers import Table


def test_should_create_json_file_from_table_by_str() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_json_file(tmp_file.name)
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_json_file(tmp_file.name)
    assert table.schema == table_r.schema
    assert table == table_r


def test_should_create_json_file_from_table_by_path() -> None:
    table = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_json_file(Path(tmp_file.name))
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_json_file(Path(tmp_file.name))
    assert table.schema == table_r.schema
    assert table == table_r
