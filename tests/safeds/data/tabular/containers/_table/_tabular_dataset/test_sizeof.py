import sys

import pytest
from safeds.data.labeled.containers import TabularDataset


@pytest.mark.parametrize(
    "tabular_dataset",
    [

            TabularDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
                ["feature_1", "feature_2"],
            ),

            TabularDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
                "target",
                ["feature_1", "feature_2"],
            ),
    ],
    ids=["normal", "table_with_column_as_non_feature"],
)
def test_should_size_be_greater_than_normal_object(tabular_dataset: TabularDataset) -> None:
    assert sys.getsizeof(tabular_dataset) > sys.getsizeof(object())
