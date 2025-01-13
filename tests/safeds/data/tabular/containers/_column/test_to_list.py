import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0],
        [0.5, 1.5],
    ],
    ids=[
        "empty",
        "one row",
        "multiple rows",
    ],
)
def test_should_return_list_of_column_values(values: list) -> None:
    column = Column("a", values)
    assert column.to_list() == values
