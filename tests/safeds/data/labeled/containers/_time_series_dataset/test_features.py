import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tabular_dataset", "features"),
    [
        (
            TimeSeriesDataset(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
                extra_names=["C"],
                window_size=1,
            ),
            Table({"A": [1, 4], "B": [2, 5]}),
        ),
        (
            TimeSeriesDataset(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                    "time": [0, 0],
                },
                target_name="T",
                window_size=1,
                extra_names=["B"],
            ),
            Table({"A": [1, 4], "C": [3, 6], "time":[0,0]}),
        ),
    ],
    ids=["only_target_and_features", "target_features_and_other"],
)
def test_should_return_features(tabular_dataset: TimeSeriesDataset, features: Table) -> None:
    assert tabular_dataset.features == features
