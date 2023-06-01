from safeds.data.tabular.containers import TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_remove_row() -> None:
    table = TaggedTable(
        {
            "feature": [0, 0, 1],
            "target": [2, 2, 3],
        },
        "target",
    )
    new_table = table.remove_duplicate_rows()
    expected = TaggedTable(
        {
            "feature": [0, 1],
            "target": [2, 3],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
