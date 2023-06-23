from safeds.data.tabular.containers import Table, TaggedTable


def test_should_return_table() -> None:
    tagged_table = TaggedTable(
        {
            "feature_1": [3, 9, 6],
            "feature_2": [6, 12, 9],
            "target": [1, 3, 2],
        },
        "target",
    )
    expected = Table(
        {
            "feature_1": [3, 9, 6],
            "feature_2": [6, 12, 9],
            "target": [1, 3, 2],
        },
    )
    table = TaggedTable.to_table(tagged_table)
    assert table.schema == expected.schema
    assert table == expected
