from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.query import ColumnSelector
from safeds.exceptions import OutOfBoundsError


@pytest.mark.parametrize(
    ("table_factory", "column_names", "z_score_threshold", "expected"),
    [
        # empty
        (
            lambda: Table({}),
            None,
            1,
            Table({}),
        ),
        # no rows
        (
            lambda: Table({"col1": []}),
            None,
            1,
            Table({"col1": []}),
        ),
        # only missing values
        (
            lambda: Table({"col1": [None, None]}),
            None,
            1,
            Table({"col1": [None, None]}),
        ),
        # no outliers (low threshold)
        (
            lambda: Table({"col1": [1, 1, 1]}),
            None,
            1,
            Table({"col1": [1, 1, 1]}),
        ),
        # no outliers (high threshold)
        (
            lambda: Table({"col1": [1, 1000]}),
            None,
            3,
            Table({"col1": [1, 1000]}),
        ),
        # outliers (all columns selected)
        (
            lambda: Table({"col1": [1, 1, 1000], "col2": [1, 1000, 1], "col3": [1000, 1, 1]}),
            None,
            1,
            Table({"col1": [], "col2": [], "col3": []}),
        ),
        # outliers (several columns selected)
        (
            lambda: Table({"col1": [1, 1, 1000], "col2": [1, 1000, 1], "col3": [1000, 1, 1]}),
            ["col1", "col2"],
            1,
            Table({"col1": [1], "col2": [1], "col3": [1000]}),
        ),
        # outliers (one column selected)
        (
            lambda: Table({"col1": [1, 1, 1000], "col2": [1, 1000, 1], "col3": [1000, 1, 1]}),
            "col1",
            1,
            Table({"col1": [1, 1], "col2": [1, 1000], "col3": [1000, 1]}),
        ),
        # outliers (no columns selected)
        (
            lambda: Table({"col1": [1, 1, 1000], "col2": [1, 1000, 1], "col3": [1000, 1, 1]}),
            [],
            1,
            Table({"col1": [1, 1, 1000], "col2": [1, 1000, 1], "col3": [1000, 1, 1]}),
        ),
        # selector
        (
            lambda: Table({"col1": [1, 1, 1000], "col2": [1, 1000, 1], "col3": [1000, 1, 1]}),
            ColumnSelector.by_name("col1"),
            1,
            Table({"col1": [1, 1], "col2": [1, 1000], "col3": [1000, 1]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "only missing values",
        "no outliers (low threshold)",
        "no outliers (high threshold)",
        "outliers (all columns selected)",
        "outliers (several columns selected)",
        "outliers (one column selected)",
        "outliers (no columns selected)",
        "selector",
    ],
)
class TestHappyPath:
    def test_should_remove_rows_with_outliers(
        self,
        table_factory: Callable[[], Table],
        column_names: str | list[str] | None,
        z_score_threshold: float,
        expected: Table,
    ) -> None:
        actual = table_factory().remove_rows_with_outliers(
            selector=column_names,
            z_score_threshold=z_score_threshold,
        )
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        column_names: str | list[str] | None,
        z_score_threshold: float,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.remove_rows_with_outliers(
            selector=column_names,
            z_score_threshold=z_score_threshold,
        )
        assert original == table_factory()


def test_should_raise_if_z_score_threshold_is_negative() -> None:
    with pytest.raises(OutOfBoundsError):
        Table({}).remove_rows_with_outliers(z_score_threshold=-1.0)
