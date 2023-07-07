import pytest
from safeds.data.tabular.containers import TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table" ,"expected"),
    [
        (
            TaggedTable(
                {
                    "feature": [0.0, None, 2.0],
                    "target": [3.0, 4.0, 5.0],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature": [0.0, 2.0],
                    "target": [3.0, 5.0],
                },
                "target",
            )
        ),
        (
            TaggedTable(
                {
                    "feature": [0.0, 1.0, 2.0],
                    "target": [3.0, 4.0, 5.0],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature": [0.0, 1.0, 2.0],
                    "target": [3.0, 4.0, 5.0],
                },
                "target",
            )
        ),
    ],
    ids=["with_missing_values", "without_missing_values"]
)
def test_should_remove_rows_with_missing_values(table: TaggedTable, expected: TaggedTable) -> None:
    new_table = table.remove_rows_with_missing_values()
    assert_that_tagged_tables_are_equal(new_table, expected)
