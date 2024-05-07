import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tabular_dataset", "extras"),
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
                time_name="C",
            ),
            Table(),
        ),
        (
            TimeSeriesDataset(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
                time_name="B",
                extra_names=["A", "C"],
            ),
            Table({"A": [1, 4], "C": [3, 6]}),
        ),
    ],
    ids=[
        "only_target_and_features",
        "target_features_and_extras",
    ],
)
def test_should_return_features(tabular_dataset: TimeSeriesDataset, extras: Table) -> None:
    assert tabular_dataset.extras == extras
