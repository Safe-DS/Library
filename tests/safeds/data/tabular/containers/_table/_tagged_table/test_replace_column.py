import pytest
from safeds.data.tabular.containers import Column, TaggedTable
from safeds.exceptions import (
    ColumnSizeError,
    DuplicateColumnNameError,
    IllegalSchemaModificationError,
    UnknownColumnNameError,
)

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("original_table", "new_columns", "column_name_to_be_replaced", "result_table"),
    [
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
            [Column("feature_new", [2, 1, 0])],
            "feature_old",
            TaggedTable(
                {
                    "feature_new": [2, 1, 0],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_new"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
            [Column("feature_new_a", [2, 1, 0]), Column("feature_new_b", [4, 2, 0])],
            "feature_old",
            TaggedTable(
                {
                    "feature_new_a": [2, 1, 0],
                    "feature_new_b": [4, 2, 0],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_new_a", "feature_new_b"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
            [Column("no_feature_new", [2, 1, 0])],
            "no_feature_old",
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_new": [2, 1, 0],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
            [Column("no_feature_new_a", [2, 1, 0]), Column("no_feature_new_b", [4, 2, 0])],
            "no_feature_old",
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_new_a": [2, 1, 0],
                    "no_feature_new_b": [4, 2, 0],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                ["feature_old"],
            ),
            [Column("target_new", [2, 1, 0])],
            "target_old",
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_new": [2, 1, 0],
                },
                "target_new",
                ["feature_old"],
            ),
        ),
    ],
    ids=[
        "replace_feature_column_with_one",
        "replace_feature_column_with_multiple",
        "replace_non_feature_column_with_one",
        "replace_non_feature_column_with_multiple",
        "replace_target_column",
    ],
)
def test_should_replace_column(
    original_table: TaggedTable,
    new_columns: list[Column],
    column_name_to_be_replaced: str,
    result_table: TaggedTable,
) -> None:
    new_table = original_table.replace_column(column_name_to_be_replaced, new_columns)
    assert_that_tagged_tables_are_equal(new_table, result_table)


@pytest.mark.parametrize(
    ("original_table", "new_columns", "column_name_to_be_replaced"),
    [
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
            ),
            [],
            "target_old",
        ),
        (
            TaggedTable(
                {
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
            ),
            [Column("target_new_a", [2, 1, 0]), Column("target_new_b"), [4, 2, 0]],
            "target_old",
        ),
    ],
    ids=["zero_columns", "multiple_columns"],
)
def test_should_throw_illegal_schema_modification(
    original_table: TaggedTable,
    new_columns: list[Column],
    column_name_to_be_replaced: str,
) -> None:
    with pytest.raises(
        IllegalSchemaModificationError,
        match='Target column "target_old" can only be replaced by exactly one new column.',
    ):
        original_table.replace_column(column_name_to_be_replaced, new_columns)


@pytest.mark.parametrize(
    ("old_column_name", "column", "error", "error_message"),
    [
        (
            "D",
            [Column("C", ["d", "e", "f"])],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D'",
        ),
        (
            "target",
            [Column("non-feat", ["d", "e", "f"]), Column("D", [3, 2, 1])],
            DuplicateColumnNameError,
            r"Column 'non-feat' already exists.",
        ),
        (
            "target",
            [Column("D", [7, 8]), Column("E", ["c", "b"])],
            ColumnSizeError,
            r"Expected a column of size 3 but got column of size 2.",
        ),
    ],
    ids=["UnknownColumnNameError", "DuplicateColumnNameError", "ColumnSizeError"],
)
def test_should_raise_error(
    old_column_name: str,
    column: list[Column],
    error: type[Exception],
    error_message: str,
) -> None:
    input_table: TaggedTable = TaggedTable(
        {
            "feat": [1, 2, 3],
            "non-feat": [4, 5, 6],
            "target": ["a", "b", "c"],
        }, "target", ["feat"]
    )

    with pytest.raises(error, match=error_message):
        input_table.replace_column(old_column_name, column)
