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
    ("index", "table"),
    [
        (-1, Table({"A": [1], "B": [2]})),
        (5, Table({"A": [1], "B": [2]})),
        (0, Table()),
    ],
    ids=["<0", "too high", "empty"],
)
def test_should_raise_error_if_index_out_of_bounds(index: int, table: Table) -> None:
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(index)


@pytest.mark.parametrize(
    ("index", "expected_error_message"),
    [(-1, r"There is no element at index '-1'."), (5, r"There is no element at index '5'.")],
    ids=["<0", "too high"],
)
def test_should_raise_error_if_index_out_of_bounds_error_message(index: int, expected_error_message: str) -> None:
    table = Table({"A": [1], "B": [2]})
    with pytest.raises(IndexOutOfBoundsError, match=expected_error_message):
        table.get_row(index)
