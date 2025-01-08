import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import RowCountMismatchError


def test_should_raise_error_if_row_counts_differ() -> None:
    with pytest.raises(RowCountMismatchError):
        Table({"a": [1, 2], "b": [3]})
