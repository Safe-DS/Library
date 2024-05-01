import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tagged_table", "features"),
    [
        (
            TabularDataset(
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
            TabularDataset(
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
    ids=["only_target_and_features", "target_features_and_other"],
)
def test_should_return_features(tagged_table: TabularDataset, features: Table) -> None:
    assert tagged_table.features == features
