import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table.from_dict({"A": [1], "B": [2]}),
         Row({"A": 1, "B": 2})),
    ],
    ids=["table with one row"],
)
def test_should_get_row(table1: Table, expected: Row) -> None:
    assert table1.get_row(0) == expected


def test_should_raise_IndexOutOfBoundsError() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(-1)


def test_should_raise_IndexOutOfBoundsError() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(5)
