from typing import Any

import pytest
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (Table(), Table(), True),
        (Table({"a": [], "b": []}), Table({"a": [], "b": []}), True),
        (Table({"col1": [1]}), Table({"col1": [1]}), True),
        (Table({"col1": [1]}), Table({"col2": [1]}), False),
        (Table({"col1": [1, 2, 3]}), Table({"col1": [1, 1, 3]}), False),
        (Table({"col1": [1, 2, 3]}), Table({"col1": ["1", "2", "3"]}), False),
    ],
    ids=[
        "empty table",
        "rowless table",
        "equal tables",
        "different column names",
        "different values",
        "different types",
    ],
)
def test_should_return_whether_two_tables_are_equal(table1: Table, table2: Table, expected: bool) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    "table",
    [Table(), Table({"col1": [1]})],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_true_if_objects_are_identical(table: Table) -> None:
    assert (table.__eq__(table)) is True


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (Table({"col1": [1]}), None),
        (Table({"col1": [1]}), Row()),
    ],
    ids=[
        "Table vs. None",
        "Table vs. Row",
    ],
)
def test_should_return_not_implemented_if_other_is_not_table(table: Table, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
