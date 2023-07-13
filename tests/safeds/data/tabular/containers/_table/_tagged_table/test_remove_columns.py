import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import ColumnIsTargetError, IllegalSchemaModificationError, UnknownColumnNameError

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
                        "non_feat_1": [2, 4, 6],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                ["feat_1"],
            ),
            [],
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
    ("table", "columns", "error", "error_msg"),
    [
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                ["feat"],
            ),
            ["target"],
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                ["feat"],
            ),
            ["non_feat", "target"],
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                ["feat"],
            ),
            ["feat"],
            IllegalSchemaModificationError,
            r"Illegal schema modification: You cannot remove every feature column.",
        ),
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                ["feat"],
            ),
            ["feat", "non_feat"],
            IllegalSchemaModificationError,
            r"Illegal schema modification: You cannot remove every feature column.",
        ),
        (
            TaggedTable._from_table(
                Table({"feat": [1, 2, 3], "non-feat": [4, 5, 6], "target": [7, 8, 9]}),
                "target",
                ["feat"],
            ),
            ["feat", "feet"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'feet'",
        ),
    ],
    ids=["remove_only_target", "remove_non_feat_and_target", "remove_all_features", "remove_non_feat_and_all_features", "remove_unknown_column"],
)
def test_should_raise_in_remove_columns(
    table: TaggedTable,
    columns: list[str],
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        table.remove_columns(columns)
