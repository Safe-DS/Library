import pytest

from safeds.data.tabular.containers import Column, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("tagged_table", "column", "expected_tagged_table"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target", None
            ),
            Column("other", [6, 7, 8]),
            TaggedTable(
                {
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                    "other": [6, 7, 8],
                },
                "target",
                ["feature_1"],
            )
        )
    ],
    ids=["add_column_as_non_feature"]
)
def test_should_add_column(tagged_table: TaggedTable, column: Column, expected_tagged_table: TaggedTable) -> None:
    assert_that_tagged_tables_are_equal(tagged_table.add_column(column), expected_tagged_table)
