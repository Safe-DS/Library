from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import SchemaError


@pytest.mark.parametrize(
    ("table_factory", "others_factory", "expected"),
    [
        (
            lambda: Table({}),
            lambda: Table({}),
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            lambda: Table({"col1": []}),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1]}),
            lambda: Table({"col1": [2]}),
            Table({"col1": [1, 2]}),
        ),
        (
            lambda: Table({"col1": [1]}),
            lambda: [
                Table({"col1": [2]}),
                Table({"col1": [3]}),
            ],
            Table({"col1": [1, 2, 3]}),
        ),
    ],
    ids=[
        "empty, empty",
        "no rows, no rows",
        "with data, with data",
        "multiple tables",
    ],
)
class TestHappyPath:
    def test_should_add_rows(
        self,
        table_factory: Callable[[], Table],
        others_factory: Callable[[], Table | list[Table]],
        expected: Table,
    ) -> None:
        actual = table_factory().add_tables_as_rows(others_factory())
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        others_factory: Callable[[], Table | list[Table]],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_tables_as_rows(others_factory())
        assert original == table_factory()

    def test_should_not_mutate_others(
        self,
        table_factory: Callable[[], Table],
        others_factory: Callable[[], Table | list[Table]],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = others_factory()
        table_factory().add_tables_as_rows(original)
        assert original == others_factory()


@pytest.mark.parametrize(
    ("table", "others"),
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
        (
            Table({"col1": []}),
            [
                Table({"col1": []}),
                Table({"col2": []}),
            ],
        ),
    ],
    ids=[
        "empty table, non-empty table",  # polars does not raise for this, so we need to check it upfront
        "too few columns",
        "too many columns",
        "different column names",
        "swapped columns",
        "different column types",
        "multiple tables",
    ],
)
def test_should_raise_if_schemas_differ(table: Table, others: Table | list[Table]) -> None:
    with pytest.raises(SchemaError):
        table.add_tables_as_rows(others)
