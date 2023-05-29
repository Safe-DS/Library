import pytest
from safeds.data.tabular.containers import TaggedTable


@pytest.mark.parametrize(
    ("original_table", "old_column_name", "new_column_name", "result_table"),
    [
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
                None,
            ),
            "feature_old",
            "feature_new",
            TaggedTable(
                {
                    "feature_new": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
                None,
            ),
        ),
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                None,
            ),
            "target_old",
            "target_new",
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "target_new": [3, 4, 5],
                },
                "target_new",
                None,
            ),
        ),
    ],
    ids=["rename_feature_column", "rename_target_column"],
)
def test_should_add_column(
    original_table: TaggedTable, old_column_name: str, new_column_name: str, result_table: TaggedTable,
) -> None:
    new_table = original_table.rename_column(old_column_name, new_column_name)
    assert new_table.schema == result_table.schema
    assert new_table.features == result_table.features
    assert new_table.target == result_table.target
    assert new_table == result_table
