import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import DuplicateColumnNameError, UnknownColumnNameError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("original_table", "old_column_name", "new_column_name", "result_table"),
    [
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                feature_names=["feature_old"],
            ),
            "feature_old",
            "feature_new",
            TaggedTable(
                {
                    "feature_new": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                feature_names=["feature_new"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                target_name="target_old",
                feature_names=["feature"],
            ),
            "target_old",
            "target_new",
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target_new": [3, 4, 5],
                },
                target_name="target_new",
                feature_names=["feature"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                feature_names=["feature"],
            ),
            "no_feature_old",
            "no_feature_new",
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "no_feature_new": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                feature_names=["feature"],
            ),
        ),
    ],
    ids=["rename_feature_column", "rename_target_column", "rename_non_feature_column"],
)
def test_should_rename_column(
    original_table: TaggedTable,
    old_column_name: str,
    new_column_name: str,
    result_table: TaggedTable,
) -> None:
    new_table = original_table.rename_column(old_column_name, new_column_name)
    assert_that_tagged_tables_are_equal(new_table, result_table)


@pytest.mark.parametrize(
    ("original_table", "old_column_name", "new_column_name", "result_table", "error_msg"),
    [
        (
            TaggedTable({"feat": [1, 2, 3], "non-feat": [4, 5, 6], "target": [7, 8, 9]}, "target", ["feat"]),
            "feet",
            "feature",
            TaggedTable({"feature": [1, 2, 3], "non-feat": [4, 5, 6], "target": [7, 8, 9]}, "target", ["feature"]),
            r"Could not find column\(s\) 'feet'",
        ),
    ],
    ids=["column_does_not_exist"],
)
def test_should_raise_if_old_column_does_not_exist(
    original_table: TaggedTable,
    old_column_name: str,
    new_column_name: str,
    result_table: TaggedTable,
    error_msg: str,
) -> None:
    with pytest.raises(UnknownColumnNameError, match=error_msg):
        assert_that_tagged_tables_are_equal(
            original_table.rename_column(old_column_name, new_column_name), result_table,
        )


@pytest.mark.parametrize(
    ("original_table", "old_column_name", "new_column_name", "result_table", "error_msg"),
    [
        (
            TaggedTable({"feat": [1, 2, 3], "non-feat": [4, 5, 6], "target": [7, 8, 9]}, "target", ["feat"]),
            "feat",
            "non-feat",
            TaggedTable({"feat": [1, 2, 3], "non-feat": [4, 5, 6], "target": [7, 8, 9]}, "target", ["feat"]),
            r"Column 'non-feat' already exists.",
        ),
    ],
    ids=["column_already_exists"],
)
def test_should_raise_if_new_column_exists_already(
    original_table: TaggedTable,
    old_column_name: str,
    new_column_name: str,
    result_table: TaggedTable,
    error_msg: str,
) -> None:
    with pytest.raises(DuplicateColumnNameError, match=error_msg):
        assert_that_tagged_tables_are_equal(
            original_table.rename_column(old_column_name, new_column_name), result_table,
        )
