import sys

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "column",
    [
        Column("col1", []),
        Column("col1", [0]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_be_larger_than_normal_object(column: Column) -> None:
    assert sys.getsizeof(column) > sys.getsizeof(object())
