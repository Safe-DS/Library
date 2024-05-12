import sys

import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [0]),
        Column("a", [0, "1"]),
    ],
    ids=[
        "empty",
        "one row",
        "multiple rows",
    ],
)
def test_should_return_size_greater_than_normal_object(column: Column) -> None:
    assert sys.getsizeof(column) > sys.getsizeof(object())
