from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from safeds.data.tabular.containers import Table


def test_to_csv_file() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1"], "col2": ["col2_1"]}))
    with NamedTemporaryFile() as tmp_table_file:
        tmp_table_file.close()
        with Path(tmp_table_file.name).open("w", encoding="utf-8") as tmp_file:
            table.to_csv_file(tmp_file.name)
        with Path(tmp_table_file.name).open("r", encoding="utf-8") as tmp_file:
            table_r = Table.from_csv_file(tmp_file.name)
    assert table == table_r
