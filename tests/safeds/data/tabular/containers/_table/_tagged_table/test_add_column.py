
from safeds.data.tabular.containers import Column, TaggedTable


def test_should_add_column() -> None:
    table = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
        None,
    )
    col = Column("feature_2", [6, 7, 8])
    new_table = table.add_column(col)
    expected = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
            "feature_2": [6, 7, 8],
        },
        "target",
        None,
    )
    assert new_table.schema == expected.schema
    assert new_table == expected
