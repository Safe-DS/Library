import pytest
from polars import from_dataframe
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table({}),
        Table({"col1": []}),
        Table({"col1": [1, 2], "col2": [3, 4]}),
    ],
    ids=[
        "empty",
        "no rows",
        "non-empty",
    ],
)
def test_should_be_able_to_restore_table_from_exchange_object(table: Table) -> None:
    exchange_object = table.__dataframe__()
    restored = Table._from_polars_data_frame(from_dataframe(exchange_object))

    assert restored == table
