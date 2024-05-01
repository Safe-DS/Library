import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tabular_dataset", "features"),
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
                extra_names=["B"],
            ),
            Table({"A": [1, 4], "C": [3, 6]}),
        ),
    ],
    ids=["only_target_and_features", "target_features_and_other"],
)
def test_should_return_features(tabular_dataset: TabularDataset, features: Table) -> None:
    assert tabular_dataset.features == features
