from safeds.data.tabular.containers import TaggedTable
from safeds.data.tabular.transformation import OneHotEncoder

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_inverse_transform_table() -> None:
    table = TaggedTable({"feat1": ["a", "b", "a"], "feat2": ["a", "b", "d"], "target": [1, 2, 3]}, "target")
    transformer = OneHotEncoder().fit(table, table.features.column_names)
    transformed_table = table.transform_table(transformer)
    expected_transformed = TaggedTable(
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
    assert_that_tagged_tables_are_equal(transformed_table, expected_transformed)
    inverse_transformed_table = transformed_table.inverse_transform_table(transformer)
    assert_that_tagged_tables_are_equal(inverse_transformed_table, table)
