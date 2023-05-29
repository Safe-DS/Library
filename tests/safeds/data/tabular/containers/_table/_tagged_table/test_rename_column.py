from safeds.data.tabular.containers import TaggedTable


def test_should_add_column() -> None:
    table = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
        None,
    )
    new_table = table.rename_column("feature_1", "feature_2")
    expected = TaggedTable(
        {
            "feature_2": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
