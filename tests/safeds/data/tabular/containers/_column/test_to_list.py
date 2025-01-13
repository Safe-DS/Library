import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0],
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_list_of_column_values(values: list) -> None:
    column = Column("a", values)
    assert column.to_list() == values
