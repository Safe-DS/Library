import pytest
from safeds.data.tabular.containers import TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature": [1.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "target",
            ),
            TaggedTable(
                {
                    "feature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "target",
            )
        ),
    ]
)
def test_should_remove_row(table: TaggedTable, expected: TaggedTable) -> None:
    new_table = table.remove_rows_with_outliers()
    assert_that_tagged_tables_are_equal(new_table, expected)
