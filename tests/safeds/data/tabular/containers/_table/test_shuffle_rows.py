import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table(),
        Table({"col1": [1, 3, 5], "col2": [2, 4, 6]}),
        Table({"col1": [1], "col2": [2]}),
    ],
    ids=["Empty table", "Table with multiple rows", "Table with one row"],
)
def test_should_shuffle_rows(table: Table) -> None:
    result_table = table.shuffle_rows()
    assert table.schema == result_table.schema
    assert table.sort_rows(lambda row1, row2: row1["col1"] - row2["col1"]) == result_table.sort_rows(lambda row1, row2: row1["col1"] - row2["col1"])
