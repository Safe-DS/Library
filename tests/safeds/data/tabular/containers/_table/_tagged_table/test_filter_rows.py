from typing import Callable

import pytest

from safeds.data.tabular.containers import TaggedTable, Row

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "expected", "query"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature_1": [3, 6],
                    "feature_2": [6, 9],
                    "target": [1, 2],
                },
                "target",
            ),
            lambda row: all(row.get_value(col) < 10 for col in row.column_names)
        ),
        (
            TaggedTable(
                {
                    "feature_1": [3, 9, 6, 2],
                    "feature_2": [6, 12, 9, 3],
                    "other": [1, 2, 3, 10],
                    "target": [1, 3, 2, 4],
                },
                "target",
                ["feature_1", "feature_2"]
            ),
            TaggedTable(
                {
                    "feature_1": [3, 6],
                    "feature_2": [6, 9],
                    "other": [1, 3],
                    "target": [1, 2],
                },
                "target",
                ["feature_1", "feature_2"]
            ),
            lambda row: all(row.get_value(col) < 10 for col in row.column_names)
        ),
(
            TaggedTable(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
            ),
            lambda row: all(row.get_value(col) < 20 for col in row.column_names)
        ),
        (
            TaggedTable(
                {
                    "feature_1": [3, 9, 6, 2],
                    "feature_2": [6, 12, 9, 3],
                    "other": [1, 2, 3, 10],
                    "target": [1, 3, 2, 4],
                },
                "target",
                ["feature_1", "feature_2"]
            ),
            TaggedTable(
                {
                    "feature_1": [3, 9, 6, 2],
                    "feature_2": [6, 12, 9, 3],
                    "other": [1, 2, 3, 10],
                    "target": [1, 3, 2, 4],
                },
                "target",
                ["feature_1", "feature_2"]
            ),
            lambda row: all(row.get_value(col) < 20 for col in row.column_names)
        )
    ],
    ids=["remove_rows_with_values_greater_9", "remove_rows_with_values_greater_9_non_feature_columns", "remove_no_rows", "remove_no_rows_non_feature_columns"]
)
def test_should_filter_rows(table: TaggedTable, expected: TaggedTable, query: Callable[[Row], bool]) -> None:
    assert_that_tagged_tables_are_equal(table.filter_rows(query), expected)
