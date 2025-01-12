from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError


@pytest.mark.parametrize(
    ("table_factory", "percentage_in_first", "shuffle", "expected_1", "expected_2"),
    [
        (
            lambda: Table({}),
            0.0,
            False,
            Table({}),
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            0.0,
            False,
            Table({"col1": []}),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3, 4]}),
            1.0,
            False,
            Table({"col1": [1, 2, 3, 4]}),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3, 4]}),
            1.0,
            True,
            Table({"col1": [4, 1, 2, 3]}),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3, 4]}),
            0.0,
            False,
            Table({"col1": []}),
            Table({"col1": [1, 2, 3, 4]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3, 4]}),
            0.0,
            True,
            Table({"col1": []}),
            Table({"col1": [4, 1, 2, 3]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3, 4]}),
            0.5,
            False,
            Table({"col1": [1, 2]}),
            Table({"col1": [3, 4]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3, 4]}),
            0.5,
            True,
            Table({"col1": [4, 1]}),
            Table({"col1": [2, 3]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "all in first, no shuffle",
        "all in first, shuffle",
        "all in second, no shuffle",
        "all in second, shuffle",
        "even split, no shuffle",
        "even split, shuffle",
    ],
)
class TestHappyPath:
    def test_should_split_rows(
        self,
        table_factory: Callable[[], Table],
        percentage_in_first: float,
        shuffle: bool,
        expected_1: Table,
        expected_2: Table,
    ) -> None:
        actual_1, actual_2 = table_factory().split_rows(percentage_in_first, shuffle=shuffle)
        assert actual_1 == expected_1
        assert actual_2 == expected_2

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        percentage_in_first: float,
        shuffle: bool,
        expected_1: Table,  # noqa: ARG002
        expected_2: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.split_rows(percentage_in_first, shuffle=shuffle)
        assert original == table_factory()


@pytest.mark.parametrize(
    "percentage_in_first",
    [
        -1.0,
        2.0,
    ],
    ids=[
        "too low",
        "too high",
    ],
)
def test_should_raise_if_percentage_in_first_is_out_of_bounds(percentage_in_first: float) -> None:
    with pytest.raises(OutOfBoundsError):
        Table({}).split_rows(percentage_in_first)
