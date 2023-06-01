from safeds.data.tabular.containers import TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_remove_row() -> None:
    table = TaggedTable(
        {
            "feature": [0.0, None, 2.0],
            "target": [3.0, 4.0, 5.0],
        },
        "target",
    )
    new_table = table.remove_rows_with_missing_values()
    expected = TaggedTable(
        {
            "feature": [0.0, 2.0],
            "target": [3.0, 5.0],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
