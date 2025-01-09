from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table_factory", "seed", "expected"),
    [
        (
            lambda: Table({}),
            42,
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            42,
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            42,
            Table({"col1": [3, 2, 1]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
            42,
            Table({"col1": [3, 2, 1], "col2": [6, 5, 4]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "one column",
        "multiple columns",
    ],
)
class TestHappyPath:
    def test_should_shuffle_rows(
        self,
        table_factory: Callable[[], Table],
        seed: int,
        expected: Table,
    ) -> None:
        actual = table_factory().shuffle_rows(seed=seed)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        seed: int,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.shuffle_rows(seed=seed)
        assert original == table_factory()
