from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "predicate", "expected"),
    [
        (Column("a", []), lambda value: value == 1, True),
        (Column("a", [1]), lambda value: value == 1, True),
        (Column("a", [1, 2]), lambda value: value == 1, False),
    ],
    ids=[
        "empty",
        "predicate always true",
        "predicate sometimes false",
    ],
)
def test_should_return_true_if_all_values_satisfy_the_predicate(
    column: Column,
    predicate: Callable,
    expected: bool,
) -> None:
    assert column.all(predicate) == expected
