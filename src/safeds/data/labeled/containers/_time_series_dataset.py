from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._config import _get_device, _init_default_device
from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.data.tabular.containers import Column, Table

from ._dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset


class TimeSeriesDataset(Dataset[Table, Column]):
    """
    A time series dataset maps feature to a target column. It can be used to train machine learning models.

    Data can be segmented into windows when loading it into the models.


    Parameters
    ----------
    data:
        The data.
    target_name:
        The name of the target column.
    window_size:
        The number of consecutive sample to use as input for prediction.
    extra_names:
        Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
        the target column are used as features.
    forecast_horizon:
        The number of time steps to predict into the future.
    continuous
        Whether or not to continue the forecast in the steps before forecast horizon.

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
    ...     window_size=1,
    ...     extra_names=["error"],
    ... )
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        data: Table | Mapping[str, Sequence[Any]],
        target_name: str,
        window_size: int,
        *,
        extra_names: list[str] | None = None,
        forecast_horizon: int = 1,
        continuous: bool = False,
    ):
        from safeds.data.tabular.containers import Table

        # Preprocess inputs
        if not isinstance(data, Table):
            data = Table(data)
        if extra_names is None:
            extra_names = []

        # Derive feature names (build the set once, since comprehensions evaluate their condition every iteration)
        non_feature_names = {target_name, *extra_names}
        feature_names = [name for name in data.column_names if name not in non_feature_names]

        # Validate inputs
        if target_name in extra_names:
            raise ValueError(f"Column '{target_name}' cannot be both target and extra.")
        if len(feature_names) == 0:
            feature_names = []

        # Set attributes
        self._table: Table = data
        self._features: Table = data.remove_columns_except(feature_names)
        self._target: Column = data.get_column(target_name)
        self._window_size: int = window_size
        self._forecast_horizon: int = forecast_horizon
        self._extras: Table = data.remove_columns_except(extra_names)
        self._continuous: bool = continuous

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
            self._window_size == other._window_size
            and self._forecast_horizon == other._forecast_horizon
            and self.target == other.target
            and self.features == other.features
            and self.extras == other.extras
        )

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this time series dataset.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(
            self.target,
            self.features,
            self.extras,
            self._window_size,
            self._forecast_horizon,
        )

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
            + sys.getsizeof(self._window_size)
            + sys.getsizeof(self._forecast_horizon)
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
    def window_size(self) -> int:
        """The number of consecutive sample to use as input for prediction."""
        return self._window_size

    @property
    def forecast_horizon(self) -> int:
        """The number of time steps to predict into the future."""
        return self._forecast_horizon

    @property
    def continuous(self) -> bool:
        """True if the time series will make a continuous prediction."""
        return self._continuous

    @property
    def extras(self) -> Table:
        """
        Additional columns of the time series dataset that are neither features nor target.

        These can be used to store additional information about instances, such as IDs.
        """
        return self._extras

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self) -> Table:
        """
        Return a new `Table` containing the feature columns, the target column and the extra columns.

        The original `TimeSeriesDataset` is not modified.

        Returns
        -------
        table:
            A table containing the feature columns, the target column and the extra columns.
        """
        return self._table

    def _into_dataloader_with_window(
        self,
        window_size: int,
        forecast_horizon: int,
        batch_size: int,
        continuous: bool = False,
    ) -> DataLoader:
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
        continuous:
            Whether or not to continue the forecast in the steps before forecast horizon.

        Raises
        ------
        OutOfBoundsError
            If window_size or forecast_horizon is below 1
        ValueError
            If the size is smaller or even than forecast_horizon + window_size

        Returns
        -------
        result:
            The DataLoader.
        """
        import torch
        from torch.utils.data import DataLoader

        _init_default_device()

        target_tensor = torch.tensor(self.target._series.to_numpy(), dtype=torch.float32)

        x_s = []
        y_s = []

        size = target_tensor.size(0)
        _check_bounds("window_size", window_size, lower_bound=_ClosedBound(1))
        _check_bounds("forecast_horizon", forecast_horizon, lower_bound=_ClosedBound(1))
        if size <= forecast_horizon + window_size:
            raise ValueError("Can not create windows with window size less then forecast horizon + window_size")
        # create feature windows and for that features targets lagged by forecast len
        # every feature column wird auch gewindowed
        # -> [i, win_size],[target]
        feature_cols = self.features.to_columns()
        for i in range(size - (forecast_horizon + window_size)):
            window = target_tensor[i : i + window_size]
            if continuous:
                label = target_tensor[i + window_size : i + window_size + forecast_horizon]

            else:
                label = target_tensor[i + window_size + forecast_horizon].unsqueeze(0)
            for col in feature_cols:
                data = torch.tensor(col._series.to_numpy(), dtype=torch.float32)
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
        OutOfBoundsError
            If window_size or forecast_horizon is below 1
        ValueError
            If the size is smaller or even than forecast_horizon + window_size

        Returns
        -------
        result:
            The DataLoader.
        """
        import torch
        from torch.utils.data import DataLoader

        _init_default_device()

        target_tensor = self.target._series.to_torch().to(_get_device())
        x_s = []

        size = target_tensor.size(0)
        _check_bounds("window_size", window_size, lower_bound=_ClosedBound(1))
        _check_bounds("forecast_horizon", forecast_horizon, lower_bound=_ClosedBound(1))
        if size <= forecast_horizon + window_size:
            raise ValueError("Can not create windows with window size less then forecast horizon + window_size")

        feature_cols = self.features.to_columns()
        for i in range(size - (forecast_horizon + window_size)):
            window = target_tensor[i : i + window_size]
            for col in feature_cols:
                data = torch.tensor(col._series.to_numpy(), dtype=torch.float32)
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


def _create_dataset(features: torch.Tensor, target: torch.Tensor) -> TorchDataset:
    from torch.utils.data import Dataset as TorchDataset

    _init_default_device()

    class _CustomDataset(TorchDataset):
        def __init__(self, features_dataset: torch.Tensor, target_dataset: torch.Tensor):
            self.X = features_dataset.float()
            self.Y = target_dataset.float()
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X[item], self.Y[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features, target)


def _create_dataset_predict(features: torch.Tensor) -> TorchDataset:
    from torch.utils.data import Dataset as TorchDataset

    _init_default_device()

    class _CustomDataset(TorchDataset):
        def __init__(self, datas: torch.Tensor):
            self.X = datas.float()
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> torch.Tensor:
            return self.X[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features)
