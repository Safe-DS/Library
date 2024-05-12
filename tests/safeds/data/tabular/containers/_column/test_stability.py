from typing import Any

import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 1),
        ([1, 2, 3, 4, None], 1 / 4),
        ([None], 1),
        ([1, 2, 3, 4], 1 / 4),
        (["b", "a", "abc", "abc", "abc"], 3 / 5),
    ],
    ids=[
        "empty",
        "some missing values",
        "only missing values",
        "numeric",
        "non-numeric",
    ],
)
def test_should_return_stability_of_column(values: list[Any], expected: float) -> None:
    column = Column("col", values)
    assert column.stability() == expected
