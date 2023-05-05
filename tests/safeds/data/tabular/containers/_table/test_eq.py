from typing import Any

import pytest
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (Table({}), Table({}), True),
        (Table.from_dict({"col1": [1]}), Table.from_dict({"col1": [1]}), True),
        (Table.from_dict({"col1": [1]}), Table.from_dict({"col2": [1]}), False),
        (Table.from_dict({"col1": [1, 2, 3]}), Table.from_dict({"col1": [1, 1, 3]}), False),
        (Table.from_dict({"col1": [1, 2, 3]}), Table.from_dict({"col1": ["1", "2", "3"]}), False),
    ],
    ids=[
        "empty Table",
        "equal Tables",
        "different column names",
        "different values",
        "different types",
    ],
)
def test_should_return_whether_two_tables_are_equal(table1: Table, table2: Table, expected: bool) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    "table",
    [
        Table.from_dict({}),
        Table.from_dict({"col1": [1]})
    ],
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
        (Table.from_dict({"col1": [1]}), None),
        (Table.from_dict({"col1": [1]}), Row()),
    ],
    ids=[
        "Table vs. None",
        "Table vs. Row",
    ],
)
def test_should_return_not_implemented_if_other_is_not_table(table: Table, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
