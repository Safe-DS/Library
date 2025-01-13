import sys

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    "table",
    [
        Table({}),
        Table({"col1": []}),
        Table({"col1": [0, 1], "col2": ["a", "b"]}),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
def test_should_size_be_greater_than_normal_object(table: Table) -> None:
    row = _LazyVectorizedRow(table)
    assert sys.getsizeof(row) > sys.getsizeof(object())
