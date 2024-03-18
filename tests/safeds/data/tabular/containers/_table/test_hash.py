from typing import Any

import pytest
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (Table(), Table()),
        (Table({"a": [], "b": []}), Table({"a": [], "b": []})),
        (Table({"col1": [1]}), Table({"col1": [1]})),
        (Table({"col1": [1, 2, 3]}), Table({"col1": [1, 1, 3]})),
    ],
    ids=[
        "empty table",
        "rowless table",
        "equal tables",
        "different values",
    ],
)
def test_should_return_same_hash_for_equal_tables(table1: Table, table2: Table) -> None:
    assert hash(table1) == hash(table2)


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (Table({"col1": [1]}), Table({"col2": [1]})),
        (Table({"col1": [1, 2, 3]}), Table({"col1": ["1", "2", "3"]})),
        (Table({"col1": [1, 2, 3]}), Table({"col1": [1, 2, 3, 4]})),
    ],
    ids=[
        "different column names",
        "different types",
        "different number of rows"
    ],
)
def test_should_return_different_hash_for_unequal_tables(table1: Table, table2: Table) -> None:
    assert hash(table1) != hash(table2)
