import pytest

from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tabular_dataset", "expected"),
    [
        (
            TimeSeriesDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
                window_size=1,
            ),
            Table(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
            ),
        ),
        (
            TimeSeriesDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
                "target",
                window_size=1,
                extra_names=["other"],
            ),
            Table(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
            ),
        ),
    ],
    ids=["normal", "table_with_extra_column"],
)
def test_should_return_table(tabular_dataset: TimeSeriesDataset, expected: Table) -> None:
    actual = tabular_dataset.to_table()
    assert actual == expected
