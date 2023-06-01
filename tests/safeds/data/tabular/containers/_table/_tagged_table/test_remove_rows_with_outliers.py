from safeds.data.tabular.containers import TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_remove_row() -> None:
    table = TaggedTable(
        {
            "feature": [1.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "target",
    )
    new_table = table.remove_rows_with_outliers()
    expected = TaggedTable(
        {
            "feature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
