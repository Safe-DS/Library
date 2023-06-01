import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import ColumnIsTaggedError


def test_should_transform_table() -> None:
    table = TaggedTable({"feat1": ["a", "b", "a"], "feat2": ["a", "b", "d"], "target": [1, 2, 3]}, "target")
    transformer = OneHotEncoder().fit(table, table.features.column_names)
    transformed_table = table.transform_table(transformer)
    expected = TaggedTable(
        {
            "feat1__a": [1.0, 0.0, 1.0],
            "feat1__b": [0.0, 1.0, 0.0],
            "feat2__a": [1.0, 0.0, 0.0],
            "feat2__b": [0.0, 1.0, 0.0],
            "feat2__d": [0.0, 0.0, 1.0],
            "target": [1, 2, 3],
        },
        "target",
    )
    assert transformed_table.schema == expected.schema
    assert transformed_table.features == expected.features
    assert transformed_table.target == expected.target
    assert transformed_table == expected


def test_should_raise_column_is_tagged() -> None:
    table = TaggedTable({"feat1": ["a", "b", "a"], "feat2": ["a", "b", "d"], "target": [1, 2, 3]}, "target")
    transformer = OneHotEncoder().fit(table, None)
    # Passing None means all columns get one-hot-encoded, i.e. also the target column!
    with pytest.raises(
        ColumnIsTaggedError,
        match='Illegal schema modification: Column "target" is tagged and cannot be removed.',
    ):
        table.transform_table(transformer)
