import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table({"A": [1], "B": [2]}), Row({"A": 1, "B": 2})),
    ],
    ids=["table with one row"],
)
def test_should_get_row(table1: Table, expected: Row) -> None:
    assert table1.get_row(0).schema == expected.schema
    assert table1.get_row(0) == expected


@pytest.mark.parametrize(
    "index",
    [-1, 5],
    ids=["<0", "too high"],
)
def test_should_raise_error_if_index_out_of_bounds(index: int) -> None:
    table = Table({"A": [1], "B": [2]})
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(index)
