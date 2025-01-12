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
            Table({}),
        ),
        (
            lambda: Table({"col1": [1, 2], "col2": [4, 5]}),
            Table({"col1": [1, 2], "col2": [4, 5]}),
        ),
        (
            lambda: Table({"col1": [1, 2], "col2": ["a", "b"]}),
            Table({"col1": [1, 2]}),
        ),
        (
            lambda: Table({"col1": ["a", "b"], "col2": ["a", "b"]}),
            Table({}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "only numeric columns",
        "some non-numeric columns",
        "only non-numeric columns",
    ],
)
class TestHappyPath:
    def test_should_remove_non_numeric_columns(
        self,
        table_factory: Callable[[], Table],
        expected: Table,
    ) -> None:
        actual = table_factory().remove_non_numeric_columns()
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.remove_non_numeric_columns()
        assert original == table_factory()
