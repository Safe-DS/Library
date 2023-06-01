from safeds.data.tabular.containers import Column, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_add_columns() -> None:
    table = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
    )
    cols = [
        Column("feature_2", [6, 7, 8]),
        Column("feature_3", [9, 6, 3]),
    ]
    new_table = table.add_columns(cols)
    expected = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
            "feature_2": [6, 7, 8],
            "feature_3": [9, 6, 3],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
