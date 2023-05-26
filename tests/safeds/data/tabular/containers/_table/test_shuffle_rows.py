import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": [1], "col2": [1]})),
    ],
    ids=["Table with identical values in rows"],
)
def test_should_shuffle_rows(table: Table) -> None:
    result_table = table.shuffle_rows()
    assert table.schema == result_table.schema
    assert table == result_table
