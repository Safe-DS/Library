from safeds.data.tabular.containers import TaggedTable


def test_should_remove_row() -> None:
    table = TaggedTable(
        {
            "feature": [1.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "target",
        None,
    )
    new_table = table.remove_rows_with_outliers()
    expected = TaggedTable(
        {
            "feature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
