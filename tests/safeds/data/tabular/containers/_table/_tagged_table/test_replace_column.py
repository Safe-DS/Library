import pytest
from safeds.data.tabular.containers import Column, TaggedTable


@pytest.mark.parametrize(
    ("original_table", "new_column", "column_name_to_be_replaced", "result_table"),
    [
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                None,
            ),
            Column("feature_new", [2, 1, 0]),
            "feature_old",
            TaggedTable(
                {
                    "feature_new": [2, 1, 0],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                None,
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                None,
            ),
            Column("target_new", [2, 1, 0]),
            "target_old",
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_new": [2, 1, 0],
                },
                "target_new",
                None,
            ),
        ),
    ],
    ids=["replace_feature_column", "replace_target_column"],
)
def test_should_replace_column(
    original_table: TaggedTable, new_column: Column, column_name_to_be_replaced: str, result_table: TaggedTable,
) -> None:
    new_table = original_table.replace_column(column_name_to_be_replaced, new_column)
    assert new_table.schema == result_table.schema
    assert new_table.features == result_table.features
    assert new_table.target == result_table.target
    assert new_table == result_table
