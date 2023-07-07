import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import ColumnIsTargetError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature_numerical": [0, 1, 2],
                    "feature_non_numerical": ["a", "b", "c"],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_numerical", "feature_non_numerical"],
            ),
            TaggedTable(
                {
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_numerical"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "non_feature_non_numerical": ["a", "b", "c"],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_numerical"],
            ),
            TaggedTable(
                {
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_numerical"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_numerical"],
            ),
            TaggedTable(
                {
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                ["feature_numerical"],
            ),
        ),
    ],
    ids=["non_numerical_feature", "non_numerical_non_feature", "all_numerical"],
)
def test_should_remove_columns_with_non_numerical_values(table: TaggedTable, expected: TaggedTable) -> None:
    new_table = table.remove_columns_with_non_numerical_values()
    assert_that_tagged_tables_are_equal(new_table, expected)


@pytest.mark.parametrize(
    "table",
    [
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "non_feature": [1, 2, 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                ["feature"],
            )
        ),
        (
            TaggedTable(
                {
                    "feature": [0, "x", 2],
                    "non_feature": [1, 2, 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                ["feature"],
            )
        ),
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "non_feature": [1, "x", 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                ["feature"],
            )
        ),
        (
            TaggedTable(
                {
                    "feature": [0, "x", 2],
                    "non_feature": [1, "x", 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                ["feature"],
            )
        ),
    ],
    ids=[
        "only_target_non_numerical",
        "also_feature_non_numerical",
        "also_non_feature_non_numerical",
        "all_non_numerical",
    ],
)
def test_should_throw_column_is_target(table: TaggedTable) -> None:
    with pytest.raises(
        ColumnIsTargetError,
        match='Illegal schema modification: Column "target" is the target column and cannot be removed.',
    ):
        table.remove_columns_with_non_numerical_values()
