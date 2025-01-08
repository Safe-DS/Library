import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import RowCountMismatchError


def test_should_raise_error_if_column_lengths_mismatch() -> None:
    with pytest.raises(RowCountMismatchError):
        Table({"a": [1, 2], "b": [3]})
