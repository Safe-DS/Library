from tempfile import NamedTemporaryFile

import pandas as pd
from safe_ds.data import Table


def test_write_and_read_json_valid():
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with open(tmp_table_file.name, "w", encoding="utf-8") as tmp_file:
            table.to_json(tmp_file.name)
        with open(tmp_table_file.name, "r", encoding="utf-8") as tmp_file:
            table_r = Table.from_json(tmp_file.name)
    assert table._data.equals(table_r._data)


def test_write_and_read_csv_valid():
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with open(tmp_table_file.name, "w", encoding="utf-8") as tmp_file:
            table.to_csv(tmp_file.name)
        with open(tmp_table_file.name, "r", encoding="utf-8") as tmp_file:
            table_r = Table.from_csv(tmp_file.name)
    assert table._data.equals(table_r._data)
