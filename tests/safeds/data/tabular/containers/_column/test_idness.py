from typing import Any

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "result"),
    [
        ([], 1),
        (["a", "b"], 1),
        (["a", "a", "a", "b"], 0.5),
        (["a", "a", "a", "a"], 0.25),
        (["a", "a", "a", None], 0.5),
    ],
    ids=[
        "empty",
        "all unique values",
        "some unique values",
        "all same values",
        "with missing values",
    ],
)
def test_should_return_idness(values: list[Any], result: float) -> None:
    column = Column("col1", values)
    assert column.idness() == result
