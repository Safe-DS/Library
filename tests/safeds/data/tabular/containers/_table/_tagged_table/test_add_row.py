import pytest
from safeds.data.tabular.containers import Row, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "row", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature": [0, 1],
                    "target": [3, 4],
                },
                "target",
            ),
            Row(
                {
                    "feature": 2,
                    "target": 5,
                },
            ),
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
            ),
        ),
    ],
    ids=["add_row"],
)
def test_should_add_row(table: TaggedTable, row: Row, expected: TaggedTable) -> None:
    assert_that_tagged_tables_are_equal(table.add_row(row), expected)
