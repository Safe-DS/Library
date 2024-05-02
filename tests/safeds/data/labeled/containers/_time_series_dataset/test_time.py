import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("tabular_dataset", "time_column"),
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
                time_name="A"
            ),
            Column("A", [1, 4]),
        ),
    ],
    ids=["time"],
)
def test_should_return_target(tabular_dataset: TimeSeriesDataset, time_column: Column) -> None:
    assert tabular_dataset.time == time_column
