import pytest
from safeds.data.tabular.containers import TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature": [0, 0, 1],
                    "target": [2, 2, 3],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature": [0, 1],
                    "target": [2, 3],
                },
                "target",
            ),
        ),
        (
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "target": [2, 2, 3],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "target": [2, 2, 3],
                },
                "target",
            ),
        ),
    ],
    ids=["with_duplicate_rows", "without_duplicate_rows"],
)
def test_should_remove_duplicate_rows(table: TaggedTable, expected: TaggedTable) -> None:
    new_table = table.remove_duplicate_rows()
    assert_that_tagged_tables_are_equal(new_table, expected)
