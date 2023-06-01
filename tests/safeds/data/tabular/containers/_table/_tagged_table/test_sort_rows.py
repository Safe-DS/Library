from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Row, TaggedTable


@pytest.mark.parametrize(
    ("table", "comparator", "expected"),
    [
        # TODO: Check that it works with an empty table
        (
            TaggedTable({"feature": [3, 2, 1], "target": [0, 0, 0]}, "target"),
            lambda row1, row2: row1["feature"] - row2["feature"],
            TaggedTable({"feature": [1, 2, 3], "target": [0, 0, 0]}, "target"),
        ),
        (
            TaggedTable({"feature": [1, 2, 3], "target": [0, 0, 0]}, "target"),
            lambda row1, row2: row1["feature"] - row2["feature"],
            TaggedTable({"feature": [1, 2, 3], "target": [0, 0, 0]}, "target"),
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
    assert table_sorted.schema == expected.schema
    assert table_sorted.features == expected.features
    assert table_sorted.target == expected.target
    assert table_sorted == expected


@pytest.mark.parametrize(
    ("table", "comparator", "table_copy"),
    [
        # TODO: Check that it works with an empty table
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
    assert table.schema == table_copy.schema
    assert table.features == table_copy.features
    assert table.target == table_copy.target
    assert table == table_copy
