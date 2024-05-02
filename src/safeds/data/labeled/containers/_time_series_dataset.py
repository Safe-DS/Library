from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    NonNumericColumnError,
    UnknownColumnNameError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import numpy as np
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset:
    """
    A time series dataset maps feature and time columns to a target column. Not like the TableDataset a TimeSeries needs
    contain one target and one time column, but can have empty features.

    Create a tabular dataset from a mapping of column names to their values.

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
        If no feature columns remains.

    Examples
    --------
    >>> from safeds.data.labeled.containers import TabularDataset
    >>> dataset = TimeSeriesDataset(
    ...     {"id": [1, 2, 3], "feature": [4, 5, 6], "target": [1, 2, 3], "error":[0,0,1]},
    ...     target_name="target",
    ...     time_name = "time",
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
        Compare two tabular datasets.

        Returns
        -------
        equals:
            'True' if features and targets are equal, 'False' otherwise.
        """
        if not isinstance(other, TimeSeriesDataset):
            return NotImplemented
        if self is other:
            return True
        return (self.target == other.target and self.features == other.features and self._table == other._table
                and self.time == other.time)

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this tabular dataset.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.target, self.features, self._table, self.time)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return (sys.getsizeof(self._target) + sys.getsizeof(self._features) + sys.getsizeof(self._table) +
                sys.getsizeof(self._time))

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
        Return a new `Table` containing the feature columns and the target column.

        The original `TabularDataset` is not modified.

        Returns
        -------
        table:
            A table containing the feature columns and the target column.
        """
        return self._table

    def _into_dataloader_with_window(self, window_size: int, forecast_horizon: int, batch_size: int) -> DataLoader:
        """
        Return a Dataloader for the data stored in this time series, used for training neural networks.

        It splits the target column into windows, uses them as feature and creates targets for the time series, by
        forecast length. The original table is not modified.

        Parameters
        ----------
        window_size:
            The size of the created windows

        forecast_horizon:
            The length of the forecast horizon, where all datapoints are collected until the given lag.

        batch_size:
            The size of data batches that should be loaded at one time.


        Returns
        -------
        result:
            The DataLoader.
        """
        import numpy as np
        from torch.utils.data import DataLoader

        target_np = self.target._data.to_numpy()

        x_s = []
        y_s = []

        size = len(target_np)
        # create feature windows and for that features targets lagged by forecast len
        # every feature column wird auch gewindowed
        # -> [i, win_size],[target]
        feature_cols = self.features.to_columns()
        for i in range(size - (forecast_horizon + window_size)):
            window = target_np[i: i + window_size]
            label = target_np[i + window_size + forecast_horizon]
            for col in feature_cols:
                data = col._data.to_numpy()
                window = np.concatenate((window, data[i: i + window_size]))
            x_s.append(window)
            y_s.append(label)

        return DataLoader(dataset=_create_dataset(np.array(x_s), np.array(y_s)), batch_size=batch_size)

    def _into_dataloader_with_window_predict(
        self,
        window_size: int,
        forecast_horizon: int,
        batch_size: int,
    ) -> DataLoader:
        """
        Return a Dataloader for the data stored in this time series, used for training neural networks.

        It splits the target column into windows, uses them as feature and creates targets for the time series, by
        forecast length. The original table is not modified.

        Parameters
        ----------
        window_size:
            The size of the created windows

        batch_size:
            The size of data batches that should be loaded at one time.


        Returns
        -------
        result:
            The DataLoader.
        """
        import numpy as np
        from torch.utils.data import DataLoader

        target_np = self.target._data.to_numpy()
        x_s = []

        size = len(target_np)
        feature_cols = self.features.to_columns()
        for i in range(size - (forecast_horizon + window_size)):
            window = target_np[i: i + window_size]
            for col in feature_cols:
                data = col._data.to_numpy()
                window = np.concatenate((window, data[i: i + window_size]))
            x_s.append(window)

        return DataLoader(dataset=_create_dataset_predict(np.array(x_s)), batch_size=batch_size)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the tabular dataset.

        Returns
        -------
        output:
            The generated HTML.
        """
        return self._table._repr_html_()

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------
    def plot_lagplot(self, lag: int) -> Image:
        """
        Plot a lagplot for the target column.

        Parameters
        ----------
        lag:
            The amount of lag used to plot

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the time series targets contains non-numerical values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import TimeSeries
        >>> table = TimeSeries({"time":[1, 2], "target": [3, 4], "feature":[2,2]}, target_name= "target", time_name="time", feature_names=["feature"], )
        >>> image = table.plot_lagplot(lag = 1)
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        if not self._target.type.is_numeric():
            raise NonNumericColumnError("This time series target contains non-numerical columns.")
        ax = pd.plotting.lag_plot(self._target._data, lag=lag)
        fig = ax.figure
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def plot_lineplot(self, x_column_name: str | None = None, y_column_name: str | None = None) -> Image:
        """

        Plot the time series target or the given column(s) as line plot.

        The function will take the time column as the default value for y_column_name and the target column as the
        default value for x_column_name.

        Parameters
        ----------
        x_column_name:
            The column name of the column to be plotted on the x-Axis, default is the time column.
        y_column_name:
            The column name of the column to be plotted on the y-Axis, default is the target column.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the time series given columns contain non-numerical values.

        UnknownColumnNameError
            If one of the given names does not exist in the table

        Examples
        --------
        >>> from safeds.data.tabular.containers import TimeSeries
        >>> table = TimeSeries({"time":[1, 2], "target": [3, 4], "feature":[2,2]}, target_name= "target", time_name="time", feature_names=["feature"], )
        >>> image = table.plot_lineplot()
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        table = self.to_table()
        intern_data = table._data
        intern_data.index.name = "index"
        if x_column_name is not None and not table.get_column(x_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        if y_column_name is None:
            y_column_name = self._target.name

        elif y_column_name not in table.column_names:
            raise UnknownColumnNameError([y_column_name])

        if x_column_name is None:
            x_column_name = self.time.name

        if not table.get_column(y_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        fig = plt.figure()
        ax = sns.lineplot(
            data=intern_data,
            x=x_column_name,
            y=y_column_name,
        )
        ax.set(xlabel=x_column_name, ylabel=y_column_name)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def plot_scatterplot(
        self,
        x_column_name: str | None = None,
        y_column_name: str | None = None,
    ) -> Image:
        """
        Plot the time series target or the given column(s) as scatter plot.

        The function will take the time column as the default value for x_column_name and the target column as the
        default value for y_column_name.

        Parameters
        ----------
        x_column_name:
            The column name of the column to be plotted on the x-Axis.
        y_column_name:
            The column name of the column to be plotted on the y-Axis.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the time series given columns contain non-numerical values.

        UnknownColumnNameError
            If one of the given names does not exist in the table

        Examples
        --------
                >>> from safeds.data.tabular.containers import TimeSeries
                >>> table = TimeSeries({"time":[1, 2], "target": [3, 4], "feature":[2,2]}, target_name= "target", time_name="time", feature_names=["feature"], )
                >>> image = table.plot_scatterplot()

        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        table = self.to_table()
        intern_data = table._data
        intern_data.index.name = "index"
        if x_column_name is not None and not table.get_column(x_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        if y_column_name is None:
            y_column_name = self._target.name
        elif y_column_name not in table.column_names:
            raise UnknownColumnNameError([y_column_name])
        if x_column_name is None:
            x_column_name = self.time.name

        if not table.get_column(y_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        fig = plt.figure()
        ax = sns.scatterplot(
            data=intern_data,
            x=x_column_name,
            y=y_column_name,
        )
        ax.set(xlabel=x_column_name, ylabel=y_column_name)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def plot_compare_time_series(self, time_series: list[TimeSeriesDataset]) -> Image:
        """
        Plot the given time series targets along the time on the x-axis.

        Parameters
        ----------
        time_series:
            A list of time series to be plotted.

        Returns
        -------
        plot:
              A plot with all the time series targets plotted by the time on the x-axis.

        Raises
        ------
        NonNumericColumnError
            if the target column contains non numerical values
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        if not self._target.type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        data = pd.DataFrame()
        data[self.time.name] = self.time._data
        data[self.target.name] = self.target._data
        for index, ts in enumerate(time_series):
            if not ts.target.type.is_numeric():
                raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")
            data[ts.target.name + " " + str(index)] = ts.target._data
        fig = plt.figure()
        data = pd.melt(data, [self.time.name])
        sns.lineplot(x=self.time.name, y="value", hue="variable", data=data)
        plt.title("Multiple Series Plot")
        plt.xlabel("Time")

        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())


def _create_dataset(features: np.array, target: np.array) -> Dataset:
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class _CustomDataset(Dataset):
        def __init__(self, features_dataset: np.array, target_dataset: np.array):
            self.X = torch.from_numpy(features_dataset.astype(np.float32))
            self.Y = torch.from_numpy(target_dataset.astype(np.float32))
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X[item], self.Y[item].unsqueeze(-1)

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features, target)


def _create_dataset_predict(features: np.array) -> Dataset:
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class _CustomDataset(Dataset):
        def __init__(self, features: np.array):
            self.X = torch.from_numpy(features.astype(np.float32))
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> torch.Tensor:
            return self.X[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features)
