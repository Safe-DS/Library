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
                    }
                ),
                "target"
            ),
            ["feat1", "target"],
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "target": [7, 8, 9],
                    },
                ),
                "target"
            )
        )
    ],
    ids=["table"],
)
def test_should_return_table(table: TaggedTable, column_names: list[str], expected: TaggedTable) -> None:
    new_table = table.keep_only_columns(column_names)
    assert_that_tagged_tables_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "column_names"),
    [
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "target": [7, 8, 9],
                    }
                ),
                "target"
            ),
            ["feat1", "feat2"],
        )
    ],
    ids=["table"],
)
def should_raise_illegal_schema_modification(table: TaggedTable, column_names: list[str]) -> None:
    with pytest.raises(IllegalSchemaModificationError, match="Must keep target column and at least one feature column."):
        table.keep_only_columns(column_names)
