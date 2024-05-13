"""Classes that can store labeled data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._dataset import Dataset
    from ._image_dataset import ImageDataset
    from ._tabular_dataset import TabularDataset
    from ._time_series_dataset import TimeSeriesDataset

apipkg.initpkg(
    __name__,
    {
        "Dataset": "._dataset:Dataset",
        "ImageDataset": "._image_dataset:ImageDataset",
        "TabularDataset": "._tabular_dataset:TabularDataset",
        "TimeSeriesDataset": "._time_series_dataset:TimeSeriesDataset",
    },
)

__all__ = [
    "Dataset",
    "ImageDataset",
    "TabularDataset",
    "TimeSeriesDataset",
]
