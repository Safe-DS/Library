from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Row, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "comparator", "expected"),
    [
        (
            TaggedTable({"feature": [3, 2, 1], "non_feature": [1, 1, 1], "target": [0, 0, 0]}, "target"),
            lambda row1, row2: row1["feature"] - row2["feature"],
            TaggedTable({"feature": [1, 2, 3], "non_feature": [1, 1, 1], "target": [0, 0, 0]}, "target"),
        ),
        (
            TaggedTable({"feature": [1, 2, 3], "non_feature": [1, 1, 1], "target": [0, 0, 0]}, "target"),
            lambda row1, row2: row1["feature"] - row2["feature"],
            TaggedTable({"feature": [1, 2, 3], "non_feature": [1, 1, 1], "target": [0, 0, 0]}, "target"),
        ),
    ],
    ids=["unsorted", "already_sorted"],
)
def test_should_sort_table(
    table: TaggedTable,
    comparator: Callable[[Row, Row], int],
    expected: TaggedTable,
) -> None:
    table_sorted = table.sort_rows(comparator)
    assert_that_tagged_tables_are_equal(table_sorted, expected)


@pytest.mark.parametrize(
    ("table", "comparator", "table_copy"),
    [
        (
            TaggedTable({"feature": [3, 2, 1], "target": [0, 0, 0]}, "target"),
            lambda row1, row2: row1["feature"] - row2["feature"],
            TaggedTable({"feature": [3, 2, 1], "target": [0, 0, 0]}, "target"),
        ),
        (
            TaggedTable({"feature": [1, 2, 3], "target": [0, 0, 0]}, "target"),
            lambda row1, row2: row1["feature"] - row2["feature"],
            TaggedTable({"feature": [1, 2, 3], "target": [0, 0, 0]}, "target"),
        ),
    ],
    ids=["unsorted", "already_sorted"],
)
def test_should_not_modify_original_table(
    table: TaggedTable,
    comparator: Callable[[Row, Row], int],
    table_copy: TaggedTable,
) -> None:
    table.sort_rows(comparator)
    assert_that_tagged_tables_are_equal(table, table_copy)
