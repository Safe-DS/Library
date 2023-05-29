from safeds.data.tabular.containers import TaggedTable, Row


def test_should_add_row() -> None:
    table = TaggedTable(
        {
            "feature": [0, 1],
            "target": [3, 4],
        },
        "target",
        None,
    )
    row = Row({
        "feature": 2,
        "target": 5,
    })
    new_table = table.add_row(row)
    expected = TaggedTable(
        {
            "feature": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
