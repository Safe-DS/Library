from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ClosedBound, OutOfBoundsError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import torch
    from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset:
    """
    A time series dataset maps feature and time columns to a target column. Not like the TabularDataset a TimeSeries needs to contain one target and one time column, but can have empty features.

    Create a time series dataset from a mapping of column names to their values.

    Parameters
    ----------
    data:
        The data.
    target_name:
        Name of the target column.
    time_name:
        Name of the time column.
    extra_names:
        Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
        the target column are used as features.

    Raises
    ------
    ColumnLengthMismatchError
        If columns have different lengths.
    ValueError
        If the target column is also an extra column.
    ValueError
        If no feature column remains.

    Examples
    --------
    >>> from safeds.data.labeled.containers import TabularDataset
    >>> dataset = TimeSeriesDataset(
    ...     {"id": [1, 2, 3], "feature": [4, 5, 6], "target": [1, 2, 3], "error":[0,0,1]},
    ...     target_name="target",
    ...     time_name = "id",
    ...     extra_names=["error"]
    ... )
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        data: Table | Mapping[str, Sequence[Any]],
        target_name: str,
        time_name: str,
        extra_names: list[str] | None = None,
    ):
        # Preprocess inputs
        if not isinstance(data, Table):
            data = Table(data)
        if extra_names is None:
            extra_names = []

        # Derive feature names
        feature_names = [name for name in data.column_names if name not in {target_name, *extra_names, time_name}]

        # Validate inputs
        if time_name in extra_names:
            raise ValueError(f"Column '{time_name}' cannot be both time and extra.")
        if target_name in extra_names:
            raise ValueError(f"Column '{target_name}' cannot be both target and extra.")
        if len(feature_names) == 0:
            feature_names = []

        # Set attributes
        self._table: Table = data
        self._features: Table = data.keep_only_columns(feature_names)
        self._target: Column = data.get_column(target_name)
        self._time: Column = data.get_column(time_name)
        self._extras: Table = data.keep_only_columns(extra_names)

    def __eq__(self, other: object) -> bool:
        """
        Compare two time series datasets.

        Returns
        -------
        equals:
            'True' if features, time, target and extras are equal, 'False' otherwise.
        """
        if not isinstance(other, TimeSeriesDataset):
            return NotImplemented
        return (self is other) or (
            self.target == other.target
            and self.features == other.features
            and self.extras == other.extras
            and self.time == other.time
        )

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this time series dataset.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.target, self.features, self.extras, self.time)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return (
            sys.getsizeof(self._target)
            + sys.getsizeof(self._features)
            + sys.getsizeof(self.extras)
            + sys.getsizeof(self._time)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def features(self) -> Table:
        """The feature columns of the time series dataset."""
        return self._features

    @property
    def target(self) -> Column:
        """The target column of the time series dataset."""
        return self._target

    @property
    def time(self) -> Column:
        """The time column of the time series dataset."""
        return self._time

    @property
    def extras(self) -> Table:
        """
        Additional columns of the time series dataset that are neither features, target nor time.

        These can be used to store additional information about instances, such as IDs.
        """
        return self._extras

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self) -> Table:
        """
        Return a new `Table` containing the feature columns, the target column, the time column and the extra columns.

        The original `TimeSeriesDataset` is not modified.

        Returns
        -------
        table:
            A table containing the feature columns, the target column, the time column and the extra columns.
        """
        return self._table

    def _into_dataloader_with_window(self, window_size: int, forecast_horizon: int, batch_size: int) -> DataLoader:
        """
        Return a Dataloader for the data stored in this time series, used for training neural networks.

        It splits the target column into windows, uses them as feature and creates targets for the time series, by
        forecast length. The original time series dataset is not modified.

        Parameters
        ----------
        window_size:
            The size of the created windows
        forecast_horizon:
            The length of the forecast horizon, where all datapoints are collected until the given lag.
        batch_size:
            The size of data batches that should be loaded at one time.

        Raises
        ------
        OutOfBoundsError:
            If window_size or forecast_horizon is below 1
        ValueError:
            If the size is smaller or even than forecast_horizon + window_size

        Returns
        -------
        result:
            The DataLoader.
        """
        import torch
        from torch.utils.data import DataLoader

        _init_default_device()

        target_tensor = torch.tensor(self.target._data.values, dtype=torch.float32)

        x_s = []
        y_s = []

        size = target_tensor.size(0)
        if window_size < 1:
            raise OutOfBoundsError(actual=window_size, name="window_size", lower_bound=ClosedBound(1))
        if forecast_horizon < 1:
            raise OutOfBoundsError(actual=forecast_horizon, name="forecast_horizon", lower_bound=ClosedBound(1))
        if size <= forecast_horizon + window_size:
            raise ValueError("Can not create windows with window size less then forecast horizon + window_size")
        # create feature windows and for that features targets lagged by forecast len
        # every feature column wird auch gewindowed
        # -> [i, win_size],[target]
        feature_cols = self.features.to_columns()
        for i in range(size - (forecast_horizon + window_size)):
            window = target_tensor[i : i + window_size]
            label = target_tensor[i + window_size + forecast_horizon]
            for col in feature_cols:
                data = torch.tensor(col._data.values, dtype=torch.float32)
                window = torch.cat((window, data[i : i + window_size]), dim=0)
            x_s.append(window)
            y_s.append(label)
        x_s_tensor = torch.stack(x_s)
        y_s_tensor = torch.stack(y_s)
        dataset = _create_dataset(x_s_tensor, y_s_tensor)
        return DataLoader(dataset=dataset, batch_size=batch_size)

    def _into_dataloader_with_window_predict(
        self,
        window_size: int,
        forecast_horizon: int,
        batch_size: int,
    ) -> DataLoader:
        """
        Return a Dataloader for the data stored in this time series, used for training neural networks.

        It splits the target column into windows, uses them as feature and creates targets for the time series, by
        forecast length. The original time series dataset is not modified.

        Parameters
        ----------
        window_size:
            The size of the created windows
        batch_size:
            The size of data batches that should be loaded at one time.

        Raises
        ------
        OutOfBoundsError:
            If window_size or forecast_horizon is below 1
        ValueError:
            If the size is smaller or even than forecast_horizon + window_size

        Returns
        -------
        result:
            The DataLoader.
        """
        import torch
        from torch.utils.data import DataLoader

        _init_default_device()

        target_tensor = torch.tensor(self.target._data.values, dtype=torch.float32)
        x_s = []

        size = target_tensor.size(0)
        if window_size < 1:
            raise OutOfBoundsError(actual=window_size, name="window_size", lower_bound=ClosedBound(1))
        if forecast_horizon < 1:
            raise OutOfBoundsError(actual=forecast_horizon, name="forecast_horizon", lower_bound=ClosedBound(1))
        if size <= forecast_horizon + window_size:
            raise ValueError("Can not create windows with window size less then forecast horizon + window_size")

        feature_cols = self.features.to_columns()
        for i in range(size - (forecast_horizon + window_size)):
            window = target_tensor[i : i + window_size]
            for col in feature_cols:
                data = torch.tensor(col._data.values, dtype=torch.float32)
                window = torch.cat((window, data[i : i + window_size]), dim=-1)
            x_s.append(window)

        x_s_tensor = torch.stack(x_s)

        dataset = _create_dataset_predict(x_s_tensor)
        return DataLoader(dataset=dataset, batch_size=batch_size)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the time series dataset.

        Returns
        -------
        output:
            The generated HTML.
        """
        return self._table._repr_html_()


def _create_dataset(features: torch.Tensor, target: torch.Tensor) -> Dataset:
    from torch.utils.data import Dataset

    _init_default_device()

    class _CustomDataset(Dataset):
        def __init__(self, features_dataset: torch.Tensor, target_dataset: torch.Tensor):
            self.X = features_dataset
            self.Y = target_dataset.unsqueeze(-1)
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X[item], self.Y[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features, target)


def _create_dataset_predict(features: torch.Tensor) -> Dataset:
    from torch.utils.data import Dataset

    _init_default_device()

    class _CustomDataset(Dataset):
        def __init__(self, features: torch.Tensor):
            self.X = features
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> torch.Tensor:
            return self.X[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features)
