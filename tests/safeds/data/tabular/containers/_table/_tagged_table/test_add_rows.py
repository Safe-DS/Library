from safeds.data.tabular.containers import Row, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_add_rows() -> None:
    table = TaggedTable(
        {
            "feature": [0, 1],
            "target": [4, 5],
        },
        "target",
    )
    rows = [
        Row(
            {
                "feature": 2,
                "target": 6,
            },
        ),
        Row({"feature": 3, "target": 7}),
    ]
    new_table = table.add_rows(rows)
    expected = TaggedTable(
        {
            "feature": [0, 1, 2, 3],
            "target": [4, 5, 6, 7],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
