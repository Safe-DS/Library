import pytest
from safeds.data.tabular.containers import Column, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("tagged_table", "columns", "expected_tagged_table"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
                None,
            ),
            [Column("other", [6, 7, 8]), Column("other2", [9, 6, 3])],
            TaggedTable(
                {
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                    "other": [6, 7, 8],
                    "other2": [9, 6, 3],
                },
                "target",
                ["feature_1"],
            ),
        ),
    ],
    ids=["add_columns_as_non_feature"],
)
def test_should_add_columns(
    tagged_table: TaggedTable, columns: list[Column], expected_tagged_table: TaggedTable,
) -> None:
    assert_that_tagged_tables_are_equal(tagged_table.add_columns(columns), expected_tagged_table)
