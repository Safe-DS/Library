import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import ColumnIsTargetError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "columns", "expected"),
    [
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                ["feat_1", "feat_2"],
            ),
            ["feat_2"],
            TaggedTable._from_table(
                Table({"feat_1": [1, 2, 3], "non_feat_1": [2, 4, 6], "non_feat_2": [3, 6, 9], "target": [7, 8, 9]}),
                "target",
                ["feat_1"],
            ),
        ),
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                ["feat_1", "feat_2"],
            ),
            ["non_feat_2"],
            TaggedTable._from_table(
                Table({"feat_1": [1, 2, 3], "feat_2": [4, 5, 6], "non_feat_1": [2, 4, 6], "target": [7, 8, 9]}),
                "target",
                ["feat_1", "feat_2"],
            ),
        ),
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                ["feat_1", "feat_2"],
            ),
            ["non_feat_1", "non_feat_2"],
            TaggedTable._from_table(
                Table({"feat_1": [1, 2, 3], "feat_2": [4, 5, 6], "target": [7, 8, 9]}),
                "target",
                ["feat_1", "feat_2"],
            ),
        ),
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                ["feat_1", "feat_2"],
            ),
            ["feat_2", "non_feat_2"],
            TaggedTable._from_table(
                Table({"feat_1": [1, 2, 3], "non_feat_1": [2, 4, 6], "target": [7, 8, 9]}),
                "target",
                ["feat_1"],
            ),
        ),
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                ["feat_1", "feat_2"],
            ),
            ["feat_2", "non_feat_2"],
            TaggedTable._from_table(
                Table({"feat_1": [1, 2, 3], "non_feat_1": [2, 4, 6], "target": [7, 8, 9]}),
                "target",
                ["feat_1"],
            ),
        ),
    ],
    ids=[
        "remove_feature",
        "remove_non_feature",
        "remove_all_non_features",
        "remove_some_feat_and_some_non_feat",
        "remove_nothing",
    ],
)
def test_should_remove_columns(table: TaggedTable, columns: list[str], expected: TaggedTable) -> None:
    new_table = table.remove_columns(columns)
    assert_that_tagged_tables_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                ["feat"],
            ),
            ["target"],
        ),
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                ["feat"],
            ),
            ["non_feat", "target"],
        ),
    ],
    ids=["remove_only_target", "remove_non_feat_and_target"],
)
def test_should_raise_column_is_target_error(table: TaggedTable, columns: list[str]) -> None:
    with pytest.raises(
        ColumnIsTargetError,
        match=r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
    ):
        table.remove_columns(columns)
