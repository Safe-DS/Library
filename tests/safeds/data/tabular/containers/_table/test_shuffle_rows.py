import numpy as np
import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), Table()),
        (Table({"col1": [1, 3, 5], "col2": [2, 4, 6]}), Table({"col1": [5, 1, 3], "col2": [6, 2, 4]})),
        (Table({"col1": [1], "col2": [2]}), Table({"col1": [1], "col2": [2]})),
    ],
    ids=["Empty table", "Table with multiple rows", "Table with one row"],
)
def test_should_shuffle_rows(table: Table, expected: Table) -> None:
    np.random.RandomState(123456)
    result_table = table.shuffle_rows()
    assert table.schema == result_table.schema
    assert result_table == expected
