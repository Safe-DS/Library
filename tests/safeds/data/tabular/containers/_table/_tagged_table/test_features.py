import pytest
from safeds.data.tabular.containers import Table, TaggedTable


@pytest.mark.parametrize(
    ("tagged_table", "features"),
    [
        (
            TaggedTable(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
            ),
            Table({"A": [1, 4], "B": [2, 5], "C": [3, 6]}),
        ),
        (
            TaggedTable(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
                feature_names=["A", "C"],
            ),
            Table({"A": [1, 4], "C": [3, 6]}),
        ),
    ],
    ids=["all_columns_are_feature", "not_all_columns_are_features"],
)
def test_should_return_features(tagged_table: TaggedTable, features: Table) -> None:
    assert tagged_table.features == features
