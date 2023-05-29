from safeds.data.tabular.containers import TaggedTable


def test_should_remove_row() -> None:
    table = TaggedTable(
        {
            "feature": [0, 0, 1],
            "target": [2, 2, 3],
        },
        "target",
        None,
    )
    new_table = table.remove_duplicate_rows()
    expected = TaggedTable(
        {
            "feature": [0, 1],
            "target": [2, 3],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
