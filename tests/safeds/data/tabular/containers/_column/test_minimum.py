import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import NonNumericColumnError


def test_minimum_invalid() -> None:
    table = Table.from_dict({"col1": ["col1_1", 2]})
    column = table.get_column("col1")
    with pytest.raises(NonNumericColumnError):
        column.minimum()


def test_minimum_valid() -> None:
    table = Table.from_dict({"col1": [1, 2, 3, 4]})
    column = table.get_column("col1")
    assert column.minimum() == 1
