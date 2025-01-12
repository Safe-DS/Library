from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table_factory", "expected"),
    [
        (
            lambda: Table({}),
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            Table({"col1": [1, 3, 2]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
            Table({"col1": [1, 3, 2], "col2": [4, 6, 5]}),
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
        expected: Table,
    ) -> None:
        actual = table_factory().shuffle_rows()
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.shuffle_rows()
        assert original == table_factory()
