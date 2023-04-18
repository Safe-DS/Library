import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import IndexOutOfBoundsError

from tests.helpers import resolve_resource_path


def test_get_row() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    val = table.get_row(0)
    assert val.get_value("A") == 1
    assert val.get_value("B") == 2


def test_get_row_negative_index() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(-1)


def test_get_row_out_of_bounds_index() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(IndexOutOfBoundsError):
        table.get_row(5)
