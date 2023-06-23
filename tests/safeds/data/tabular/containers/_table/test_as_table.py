import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table(
            {
                "col1": [1, 2],
                "col2:": [3, 4]
            }
        ),
        Table()
    ],
    ids=["table", "empty"]
)
def should_return_table(table: Table) -> None:
    new_table = table._as_table()
    assert table.schema == new_table.schema
    assert table == new_table
