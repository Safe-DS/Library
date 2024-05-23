import pytest
from safeds.data.labeled.containers import TimeSeriesDataset


@pytest.mark.parametrize(
    ("tabular_dataset", "continuous"),
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
                window_size=1,
                continuous=True,
            ),
            True,
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
                window_size=1,
                extra_names=["A", "C"],
                continuous=False,
            ),
            False,
        ),
    ],
    ids=[
        "true pre",
        "target_features_and_extras",
    ],
)
def test_should_return_features(tabular_dataset: TimeSeriesDataset, continuous: bool) -> None:
    assert tabular_dataset.continuous == continuous
