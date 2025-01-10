from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import SchemaError


@pytest.mark.parametrize(
    ("table_factory", "other", "expected"),
    [
        (
            lambda: Table({}),
            Table({}),
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            Table({"col1": []}),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1]}),
            Table({"col1": [2]}),
            Table({"col1": [1, 2]}),
        ),
    ],
    ids=[
        "empty, empty",
        "no rows, no rows",
        "with data, with data",
    ],
)
class TestHappyPath:
    def test_should_add_rows(
        self,
        table_factory: Callable[[], Table],
        other: Table,
        expected: Table,
    ) -> None:
        actual = table_factory().add_table_as_rows(other)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        other: Table,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_table_as_rows(other)
        assert original == table_factory()


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (
            Table({}),
            Table({"col1": [1]}),
        ),
        (
            Table({"col1": [], "col2": []}),
            Table({"col1": []}),
        ),
        (
            Table({"col1": []}),
            Table({"col1": [], "col2": []}),
        ),
        (
            Table({"col1": []}),
            Table({"col2": []}),
        ),
        (
            Table({"col1": [], "col2": []}),
            Table({"col2": [], "col1": []}),
        ),
        (
            Table({"col1": [1]}),
            Table({"col1": ["a"]}),
        ),
    ],
    ids=[
        "empty table, non-empty table",  # polars does not raise for this, so we need to check it upfront
        "too few columns",
        "too many columns",
        "different column names",
        "swapped columns",
        "different column types",
    ],
)
def test_should_raise_if_schemas_differ(table: Table, other: Table) -> None:
    with pytest.raises(SchemaError):
        table.add_table_as_rows(other)
