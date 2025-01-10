from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError


@pytest.mark.parametrize(
    ("table_factory", "start", "length", "expected"),
    [
        (
            lambda: Table({}),
            0,
            None,
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            0,
            None,
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            0,
            None,
            Table({"col1": [1, 2, 3]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            1,
            None,
            Table({"col1": [2, 3]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            10,
            None,
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            -1,
            None,
            Table({"col1": [3]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            -10,
            None,
            Table({"col1": [1, 2, 3]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            0,
            1,
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            0,
            10,
            Table({"col1": [1, 2, 3]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "full table",
        "positive start in bounds",
        "positive start out of bounds",
        "negative start in bounds",
        "negative start out of bounds",
        "positive length in bounds",
        "positive length out of bounds",
    ],
)
class TestHappyPath:
    def test_should_slice_rows(
        self,
        table_factory: Callable[[], Table],
        start: int,
        length: int | None,
        expected: Table,
    ) -> None:
        actual = table_factory().slice_rows(start=start, length=length)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        start: int,
        length: int | None,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.slice_rows(start=start, length=length)
        assert original == table_factory()


def test_should_raise_for_negative_length() -> None:
    table: Table = Table({})
    with pytest.raises(OutOfBoundsError):
        table.slice_rows(length=-1)
