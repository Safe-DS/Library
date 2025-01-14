from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column_factory", "new_name", "expected"),
    [
        (
            lambda: Column("col1", []),
            "col2",
            Column("col2", []),
        ),
        (
            lambda: Column("col1", [0]),
            "col2",
            Column("col2", [0]),
        ),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
class TestHappyPath:
    def test_should_rename_column(
        self,
        column_factory: Callable[[], Column],
        new_name: str,
        expected: Column,
    ) -> None:
        actual = column_factory().rename(new_name)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        column_factory: Callable[[], Column],
        new_name: str,
        expected: Column,  # noqa: ARG002
    ) -> None:
        original = column_factory()
        original.rename(new_name)
        assert original == column_factory()
