from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "predicate", "use_kleene_logic", "expected"),
    [
        (
            Column("a", []),
            lambda value: value == 1,
            False,
            True,
        ),
        (
            Column("a", [None, None]),
            lambda value: value == 1,
            False,
            False,
        ),
        (
            Column("a", [None, None]),
            lambda value: value == None,  # noqa: E711
            False,
            True,
        ),
        (
            Column("a", [1, 1, None]),
            lambda value: value < 2,
            False,
            True,
        ),
        (
            Column("a", [1, 1, None]),
            lambda value: value < 2,
            True,
            None,
        ),
        (
            Column("a", [1]),
            lambda value: value == 1,
            False,
            True,
        ),
        (
            Column("a", [1, 2]),
            lambda value: value == 1,
            False,
            False,
        ),
    ],
    ids=[
        "empty",
        "only missing (compared to int)",
        "only missing (compared to None)",
        "some missing (boolean logic)",
        "some missing (Kleene logic)",
        "predicate always true",
        "predicate sometimes false",
    ],
)
def test_should_return_true_if_all_values_satisfy_the_predicate(
    column: Column,
    predicate: Callable,
    use_kleene_logic: bool,
    expected: bool | None,
) -> None:
    assert column.all(predicate, use_kleene_logic=use_kleene_logic) == expected
