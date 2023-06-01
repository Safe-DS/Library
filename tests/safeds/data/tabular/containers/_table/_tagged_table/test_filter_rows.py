from safeds.data.tabular.containers import TaggedTable


def test_should_remove_row() -> None:
    table = TaggedTable(
        {
            "feature_1": [3, 9, 6],
            "feature_2": [6, 12, 9],
            "target": [1, 3, 2],
        },
        "target",
    )
    new_table = table.filter_rows(lambda row: all(row.get_value(col) < 10 for col in table.column_names))
    expected = TaggedTable(
        {
            "feature_1": [3, 6],
            "feature_2": [6, 9],
            "target": [1, 2],
        },
        "target",
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected
