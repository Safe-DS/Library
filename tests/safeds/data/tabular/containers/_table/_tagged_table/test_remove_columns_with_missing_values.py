from safeds.data.tabular.containers import TaggedTable


def test_should_remove_column() -> None:
    table = TaggedTable(
        {
            "feature_complete": [0, 1, 2],
            "feature_incomplete": [3, None, 5],
            "target": [6, 7, 8],
        },
        "target",
        None,
    )
    new_table = table.remove_columns_with_missing_values()
    expected = TaggedTable(
        {
            "feature_complete": [0, 1, 2],
            "target": [6, 7, 8],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
