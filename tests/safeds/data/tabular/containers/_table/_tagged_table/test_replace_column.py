import pytest
from safeds.data.tabular.containers import Column, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("original_table", "new_columns", "column_name_to_be_replaced", "result_table"),
    # TODO: Add multicolumn cases, add illegal cases
    [
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
            ),
            [Column("feature_new", [2, 1, 0])],
            "feature_old",
            TaggedTable(
                {
                    "feature_new": [2, 1, 0],
                    "target_old": [3, 4, 5],
                },
                "target_old",
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
            ),
            [Column("target_new", [2, 1, 0])],
            "target_old",
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_new": [2, 1, 0],
                },
                "target_new",
            ),
        ),
    ],
    ids=["replace_feature_column", "replace_target_column"],
)
def test_should_replace_column(
    original_table: TaggedTable,
    new_columns: list[Column],
    column_name_to_be_replaced: str,
    result_table: TaggedTable,
) -> None:
    new_table = original_table.replace_column(column_name_to_be_replaced, new_columns)
    assert_that_tagged_tables_are_equal(new_table, result_table)
