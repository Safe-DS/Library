from safeds.data.tabular.containers import TaggedTable


def test_should_shuffle_rows() -> None:
    table = TaggedTable(
        {
            "feature_a": [0, 1, 2],
            "feature_b": [3, 4, 5],
            "target": [6, 7, 8],
        },
        "target",
    )
    shuffled = table.shuffle_rows()
    assert table.schema == shuffled.schema
    assert table.features.column_names == shuffled.features.column_names
    assert table.target.name == shuffled.target.name
    # Use filter_rows to extract the individual rows and compare them one-by-one:
    row_0 = shuffled.filter_rows(lambda row: any(row.get_value(col) == 0 for col in table.column_names))
    row_1 = shuffled.filter_rows(lambda row: any(row.get_value(col) == 1 for col in table.column_names))
    row_2 = shuffled.filter_rows(lambda row: any(row.get_value(col) == 2 for col in table.column_names))
    expected_0 = TaggedTable(
        {
            "feature_a": [0],
            "feature_b": [3],
            "target": [6],
        },
        "target",
    )
    expected_1 = TaggedTable(
        {
            "feature_a": [1],
            "feature_b": [4],
            "target": [7],
        },
        "target",
    )
    expected_2 = TaggedTable(
        {
            "feature_a": [2],
            "feature_b": [5],
            "target": [8],
        },
        "target",
    )
    assert row_0 == expected_0
    assert row_1 == expected_1
    assert row_2 == expected_2
