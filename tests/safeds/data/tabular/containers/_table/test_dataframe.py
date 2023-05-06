import pytest
from pandas.core.interchange.from_dataframe import from_dataframe
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table([]),
        Table.from_dict({"a": [1, 2], "b": [3, 4]}),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_restore_table_from_exchange_object(table: Table) -> None:
    exchange_object = table.__dataframe__()
    restored = Table(from_dataframe(exchange_object))

    assert restored == table


def test_should_raise_error_if_allow_copy_is_false() -> None:
    table = Table.from_dict({})
    with pytest.raises(NotImplementedError, match="`allow_copy` must be True"):
        table.__dataframe__(allow_copy=False)
