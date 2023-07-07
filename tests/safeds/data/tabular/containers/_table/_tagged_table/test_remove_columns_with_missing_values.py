import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import ColumnIsTargetError, IllegalSchemaModificationError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature_complete": [0, 1, 2],
                    "feature_incomplete": [3, None, 5],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_complete", "feature_incomplete"],
            ),
            TaggedTable(
                {
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_complete"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "non_feature_incomplete": [3, None, 5],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_complete"],
            ),
            TaggedTable(
                {
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_complete"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_complete"],
            ),
            TaggedTable(
                {
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_complete"],
            ),
        ),
    ],
    ids=["incomplete_feature", "incomplete_non_feature", "all_complete"],
)
def test_should_remove_columns_with_non_numerical_values(table: TaggedTable, expected: TaggedTable) -> None:
    new_table = table.remove_columns_with_missing_values()
    assert_that_tagged_tables_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "error", "error_msg"),
    [
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "non_feature": [1, 2, 3],
                    "target": [3, None, 5],
                },
                "target",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.'
        ),
        (
            TaggedTable(
                {
                    "feature": [0, None, 2],
                    "non_feature": [1, 2, 3],
                    "target": [None, 4, 5],
                },
                "target",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.'
        ),
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "non_feature": [1, None, 3],
                    "target": [3, 4, None],
                },
                "target",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.'
        ),
        (
            TaggedTable(
                {
                    "feature": [0, None, 2],
                    "non_feature": [1, None, 3],
                    "target": [3, None, 5],
                },
                "target",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.'
        ),
        (
            TaggedTable(
                {
                    "feature": [0, None, 2],
                    "non_feature": [1, 2, 3],
                    "target": [3, 2, 5],
                },
                "target",
                ["feature"],
            ),
            IllegalSchemaModificationError,
            'Illegal schema modification: You cannot remove every feature column.'
        ),
        (
            TaggedTable(
                {
                    "feature": [0, None, 2],
                    "non_feature": [1, None, 3],
                    "target": [3, 2, 5],
                },
                "target",
                ["feature"],
            ),
            IllegalSchemaModificationError,
            'Illegal schema modification: You cannot remove every feature column.'
        ),
    ],
    ids=[
        "only_target_incomplete",
        "also_feature_incomplete",
        "also_non_feature_incomplete",
        "all_incomplete",
        "all_features_incomplete",
        "all_features_and_non_feature_incomplete"
    ],
)
def test_should_raise_in_remove_columns_with_missing_values(table: TaggedTable, error: type[Exception], error_msg: str) -> None:
    with pytest.raises(
        error,
        match=error_msg,
    ):
        table.remove_columns_with_missing_values()
