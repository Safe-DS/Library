from safeds.data.tabular.containers import TaggedTable, Row


def test_should_add_rows() -> None:
    table = TaggedTable(
        {
            "feature": [0, 1],
            "target": [4, 5],
        },
        "target",
        None,
    )
    rows = [
        Row({
            "feature": 2,
            "target": 6,
        }),
        Row({
            "feature": 3,
            "target": 7
        })
    ]
    new_table = table.add_rows(rows)
    expected = TaggedTable(
        {
            "feature": [0, 1, 2, 3],
            "target": [4, 5, 6, 7],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
