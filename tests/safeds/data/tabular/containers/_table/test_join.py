from collections.abc import Callable
from typing import Literal

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError, DuplicateColumnError, LengthMismatchError


@pytest.mark.parametrize(
    ("left_table_factory", "right_table_factory", "left_names", "right_names", "mode", "expected"),
    [
        # inner join (empty)
        (
            lambda: Table({"a": [], "b": []}),
            lambda: Table({"c": [], "d": []}),
            "a",
            "c",
            "inner",
            Table({"a": [], "b": [], "d": []}),
        ),
        # inner join (missing values only)
        (
            lambda: Table({"a": [None], "b": [None]}),
            lambda: Table({"c": [None], "d": [None]}),
            "a",
            "c",
            "inner",
            Table({"a": [], "b": [], "d": []}),
        ),
        # inner join (with data)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            "a",
            "c",
            "inner",
            Table({"a": [1], "b": [True], "d": [True]}),
        ),
        # inner join (two columns to join on)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            ["a", "b"],
            ["c", "d"],
            "inner",
            Table({"a": [1], "b": [True]}),
        ),
        # left join (empty)
        (
            lambda: Table({"a": [], "b": []}),
            lambda: Table({"c": [], "d": []}),
            "a",
            "c",
            "left",
            Table({"a": [], "b": [], "d": []}),
        ),
        # left join (missing values only)
        (
            lambda: Table({"a": [None], "b": [None]}),
            lambda: Table({"c": [None], "d": [None]}),
            "a",
            "c",
            "left",
            Table({"a": [None], "b": [None], "d": [None]}),
        ),
        # left join (with data)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            "a",
            "c",
            "left",
            Table({"a": [1, 2], "b": [True, False], "d": [True, None]}),
        ),
        # left join (two columns to join on)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            ["a", "b"],
            ["c", "d"],
            "left",
            Table({"a": [1, 2], "b": [True, False]}),
        ),
        # right join (empty)
        (
            lambda: Table({"a": [], "b": []}),
            lambda: Table({"c": [], "d": []}),
            "a",
            "c",
            "right",
            Table({"b": [], "c": [], "d": []}),
        ),
        # right join (missing values only)
        (
            lambda: Table({"a": [None], "b": [None]}),
            lambda: Table({"c": [None], "d": [None]}),
            "a",
            "c",
            "right",
            Table({"b": [None], "c": [None], "d": [None]}),
        ),
        # right join (with data)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            "a",
            "c",
            "right",
            Table({"b": [True, None], "c": [1, 3], "d": [True, True]}),
        ),
        # right join (two columns to join on)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            ["a", "b"],
            ["c", "d"],
            "right",
            Table({"c": [1, 3], "d": [True, True]}),
        ),
        # full join (empty)
        (
            lambda: Table({"a": [], "b": []}),
            lambda: Table({"c": [], "d": []}),
            "a",
            "c",
            "full",
            Table({"a": [], "b": [], "d": []}),
        ),
        # full join (missing values only)
        (
            lambda: Table({"a": [None], "b": [None]}),
            lambda: Table({"c": [None], "d": [None]}),
            "a",
            "c",
            "full",
            # nulls do not match each other
            Table({"a": [None, None], "b": [None, None], "d": [None, None]}),
        ),
        # full join (with data)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            "a",
            "c",
            "full",
            Table({"a": [1, 2, 3], "b": [True, False, None], "d": [True, None, True]}),
        ),
        # full join (two columns to join on)
        (
            lambda: Table({"a": [1, 2], "b": [True, False]}),
            lambda: Table({"c": [1, 3], "d": [True, True]}),
            ["a", "b"],
            ["c", "d"],
            "full",
            Table({"a": [1, 2, 3], "b": [True, False, True]}),
        ),
    ],
    ids=[
        "inner join (empty)",
        "inner join (missing values only)",
        "inner join (with data)",
        "inner join (two columns to join on)",
        "left join (empty)",
        "left join (missing values only)",
        "left join (with data)",
        "left join (two columns to join on)",
        "right join (empty)",
        "right join (missing values only)",
        "right join (with data)",
        "right join (two columns to join on)",
        "full join (empty)",
        "full join (missing values only)",
        "full join (with data)",
        "full join (two columns to join on)",
    ],
)
class TestHappyPath:
    def test_should_join_two_tables(
        self,
        left_table_factory: Callable[[], Table],
        right_table_factory: Callable[[], Table],
        left_names: str | list[str],
        right_names: str | list[str],
        mode: Literal["inner", "left", "right", "full"],
        expected: Table,
    ) -> None:
        actual = left_table_factory().join(right_table_factory(), left_names, right_names, mode=mode)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        left_table_factory: Callable[[], Table],
        right_table_factory: Callable[[], Table],
        left_names: str | list[str],
        right_names: str | list[str],
        mode: Literal["inner", "left", "right", "full"],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = left_table_factory()
        original.join(right_table_factory(), left_names, right_names, mode=mode)
        assert original == left_table_factory()

    def test_should_not_mutate_right_table(
        self,
        left_table_factory: Callable[[], Table],
        right_table_factory: Callable[[], Table],
        left_names: str | list[str],
        right_names: str | list[str],
        mode: Literal["inner", "left", "right", "full"],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = right_table_factory()
        left_table_factory().join(original, left_names, right_names, mode=mode)
        assert original == right_table_factory()


@pytest.mark.parametrize(
    ("left_table", "right_table", "left_names", "right_names"),
    [
        (
            Table({"a": [], "b": []}),
            Table({"c": [], "d": []}),
            "unknown",
            "c",
        ),
        (
            Table({"a": [], "b": []}),
            Table({"c": [], "d": []}),
            "a",
            "unknown",
        ),
    ],
    ids=[
        "wrong left name",
        "wrong right name",
    ],
)
def test_should_raise_if_columns_do_not_exist(
    left_table: Table,
    right_table: Table,
    left_names: list[str],
    right_names: list[str],
) -> None:
    with pytest.raises(ColumnNotFoundError):
        left_table.join(right_table, left_names=left_names, right_names=right_names)


@pytest.mark.parametrize(
    ("left_table", "right_table", "left_names", "right_names"),
    [
        (
            Table({"a": [], "b": []}),
            Table({"c": [], "d": []}),
            ["a", "a"],
            ["c", "d"],
        ),
        (
            Table({"a": [], "b": []}),
            Table({"c": [], "d": []}),
            ["a", "b"],
            ["c", "c"],
        ),
    ],
    ids=[
        "duplicate left name",
        "duplicate right name",
    ],
)
def test_should_raise_if_columns_are_not_unique(
    left_table: Table,
    right_table: Table,
    left_names: list[str],
    right_names: list[str],
) -> None:
    with pytest.raises(DuplicateColumnError):
        left_table.join(right_table, left_names=left_names, right_names=right_names)


def test_should_raise_if_number_of_columns_to_join_on_differs() -> None:
    left_table = Table({"a": [], "b": []})
    right_table = Table({"c": [], "d": []})
    with pytest.raises(LengthMismatchError, match="The number of columns to join on must be the same in both tables."):
        left_table.join(right_table, ["a"], ["c", "d"])


def test_should_raise_if_columns_to_join_on_are_empty() -> None:
    left_table = Table({"a": []})
    right_table = Table({"b": []})
    with pytest.raises(ValueError, match="The columns to join on must not be empty."):
        left_table.join(right_table, [], [])
