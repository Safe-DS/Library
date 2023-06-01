from safeds.data.tabular.containers import TaggedTable


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
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
