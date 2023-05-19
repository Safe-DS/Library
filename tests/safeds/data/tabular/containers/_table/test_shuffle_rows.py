import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": [1], "col2": [1]})),
        Table()
    ],
    ids=["Table with identical values in rows", "empty"],
)
def test_should_shuffle_rows(table: Table) -> None:
    result_table = table.shuffle_rows()
    assert table == result_table
