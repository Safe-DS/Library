import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import IllegalSchemaModificationError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "column_names", "expected"),
    [
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
            ),
            ["feat1", "target"],
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
            ),
        ),
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 4, 5],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
            ),
            ["feat1", "other", "target"],
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "other": [3, 4, 5],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
            ),
        ),
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 4, 5],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
            ),
            ["feat1", "target"],
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
            ),
        ),
    ],
    ids=["table", "table_keep_non_feature_column", "table_not_keep_non_feature_column"],
)
def test_should_return_table(table: TaggedTable, column_names: list[str], expected: TaggedTable) -> None:
    new_table = table.keep_only_columns(column_names)
    assert_that_tagged_tables_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "column_names", "error_msg"),
    [
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 5, 7],
                        "target": [7, 8, 9],
                    },
                ),
                "target", ["feat1", "feat2"]
            ),
            ["feat1", "feat2"],
            r"Illegal schema modification: Must keep the target column."
        ),(
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 5, 7],
                        "target": [7, 8, 9],
                    },
                ),
                "target", ["feat1", "feat2"]
            ),
            ["target", "other"],
            r"Illegal schema modification: Must keep at least one feature column."
        ),
    ],
    ids=["table_remove_target", "table_remove_all_features"],
)
def test_should_raise_illegal_schema_modification(table: TaggedTable, column_names: list[str], error_msg: str) -> None:
    with pytest.raises(
        IllegalSchemaModificationError,
        match=error_msg,
    ):
        table.keep_only_columns(column_names)
