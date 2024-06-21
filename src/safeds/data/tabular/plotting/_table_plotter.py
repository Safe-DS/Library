from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from safeds._utils import _figure_to_image
from safeds._validation import _check_columns_exist
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.exceptions import ColumnTypeError, NonNumericColumnError

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import Table


class TablePlotter:
    """
    A class that contains plotting methods for a table.

    Parameters
    ----------
    table:
        The table to plot.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table
    >>> table = Table({"test": [1, 2, 3]})
    >>> plotter = table.plot
    """

    def __init__(self, table: Table):
        self._table: Table = table

    def box_plots(self) -> Image:
        """
        Plot a boxplot for every numerical column.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the table contains only non-numerical columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a":[1, 2], "b": [3, 42]})
        >>> image = table.plot.box_plots()
        """
        # TODO: implement using matplotlib and polars
        import matplotlib.pyplot as plt
        import seaborn as sns

        numerical_table = self._table.remove_non_numeric_columns()
        if numerical_table.column_count == 0:
            raise NonNumericColumnError("This table contains only non-numerical columns.")
        col_wrap = min(numerical_table.column_count, 3)

        data = numerical_table._lazy_frame.melt(value_vars=numerical_table.column_names).collect()
        grid = sns.FacetGrid(data, col="variable", col_wrap=col_wrap, sharex=False, sharey=False)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Using the boxplot function without specifying `order` is likely to produce an incorrect plot.",
            )
            grid.map(sns.boxplot, "variable", "value")
        grid.set_xlabels("")
        grid.set_ylabels("")
        grid.set_titles("{col_name}")
        for axes in grid.axes.flat:
            axes.set_xticks([])
        plt.tight_layout()
        fig = grid.fig

        return _figure_to_image(fig)

    def correlation_heatmap(self) -> Image:
        """
        Plot a correlation heatmap for all numerical columns of this `Table`.

        Returns
        -------
        plot:
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
        >>> image = table.plot.correlation_heatmap()
        """
        # TODO: implement using matplotlib and polars
        #  https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
        import matplotlib.pyplot as plt
        import numpy as np

        only_numerical = self._table.remove_non_numeric_columns()._data_frame.fill_null(0)

        if self._table.row_count == 0:
            warnings.warn(
                "An empty table has been used. A correlation heatmap on an empty table will show nothing.",
                stacklevel=2,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Attempting to set identical low and high (xlims|ylims) makes transformation singular;"
                    " automatically expanding."
                ),
            )

            fig, ax = plt.subplots()
            heatmap = plt.imshow(
                only_numerical.corr().to_numpy(),
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
            )
            ax.set_xticks(np.arange(len(only_numerical.columns)), labels=only_numerical.columns)
            ax.set_yticks(np.arange(len(only_numerical.columns)), labels=only_numerical.columns)
            fig.colorbar(heatmap)

            plt.tight_layout()

        return _figure_to_image(fig)

    def histograms(self, *, max_bin_count: int = 10) -> Image:
        """
        Plot a histogram for every column.

        Parameters
        ----------
        max_bin_count:
            The maximum number of bins to use in the histogram. Default is 10.

        Returns
        -------
        plot:
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [2, 3, 5, 1], "b": [54, 74, 90, 2014]})
        >>> image = table.plot.histograms()
        """
        import matplotlib.pyplot as plt
        import polars as pl

        n_cols = min(3, self._table.column_count)
        n_rows = 1 + (self._table.column_count - 1) // n_cols

        if n_cols == 1 and n_rows == 1:
            fig, axs = plt.subplots(1, 1, tight_layout=True)
            one_col = True
        else:
            fig, axs = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(n_cols * 3, n_rows * 3))
            one_col = False

        col_names = self._table.column_names
        for col_name, ax in zip(col_names, axs.flatten() if not one_col else [axs], strict=False):
            column = self._table.get_column(col_name)
            distinct_values = column.get_distinct_values()

            ax.set_title(col_name)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if column.is_numeric and len(distinct_values) > max_bin_count:
                min_val = (column.min() or 0) - 1e-6  # Otherwise the minimum value is not included in the first bin
                max_val = column.max() or 0
                bin_count = min(max_bin_count, len(distinct_values))
                bins = [
                    *(pl.Series(range(bin_count + 1)) / bin_count * (max_val - min_val) + min_val),
                ]

                bars = [f"{round((bins[i] + bins[i + 1]) / 2, 2)}" for i in range(len(bins) - 1)]
                hist = column._series.hist(bins=bins).slice(1, length=max_bin_count).get_column("count").to_numpy()

                ax.bar(bars, hist, edgecolor="black")
                ax.set_xticks(range(len(hist)), bars, rotation=45, horizontalalignment="right")
            else:
                value_counts = (
                    column._series.drop_nulls().value_counts().sort(column.name).slice(0, length=max_bin_count)
                )
                distinct_values = value_counts.get_column(column.name).cast(pl.String).to_numpy()
                hist = value_counts.get_column("count").to_numpy()
                ax.bar(distinct_values, hist, edgecolor="black")
                ax.set_xticks(range(len(distinct_values)), distinct_values, rotation=45, horizontalalignment="right")

        for i in range(len(col_names), n_rows * n_cols):
            fig.delaxes(axs.flatten()[i])  # Remove empty subplots

        return _figure_to_image(fig)

    def line_plot(self, x_name: str, y_names: list[str], show_confidence_interval: bool = True) -> Image:
        """
        Create a line plot for two columns in the table.

        Parameters
        ----------
        x_name:
            The name of the column to be plotted on the x-axis.
        y_names:
            The name(s) of the column(s) to be plotted on the y-axis.
        show_confidence_interval:
            If the confidence interval is shown, per default True.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        ColumnNotFoundError
            If a column does not exist.
        TypeError
            If a column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": [2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> image = table.plot.line_plot("a", ["b"])
        """
        _plot_validation(self._table, x_name, y_names)

        import matplotlib.pyplot as plt
        import polars as pl

        agg_list = []
        for name in y_names:
            agg_list.append(pl.col(name).mean().alias(f"{name}_mean"))
            agg_list.append(pl.count(name).alias(f"{name}_count"))
            agg_list.append(pl.std(name, ddof=0).alias(f"{name}_std"))
        grouped = self._table._lazy_frame.sort(x_name).group_by(x_name).agg(agg_list).collect()

        x = grouped.get_column(x_name)
        y_s = []
        confidence_intervals = []
        for name in y_names:
            y_s.append(grouped.get_column(name + "_mean"))
            confidence_intervals.append(
                1.96 * grouped.get_column(name + "_std") / grouped.get_column(name + "_count").sqrt(),
            )

        fig, ax = plt.subplots()
        for name, y in zip(y_names, y_s, strict=False):
            ax.plot(x, y, label=name)

        if show_confidence_interval:
            for y, conf in zip(y_s, confidence_intervals, strict=False):
                ax.fill_between(
                    x,
                    y - conf,
                    y + conf,
                    color="lightblue",
                    alpha=0.15,
                )
        if len(y_names) > 1:
            name = "values"
        else:
            name = y_names[0]
        ax.set(
            xlabel=x_name,
            ylabel=name,
        )
        ax.legend()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        fig.tight_layout()

        return _figure_to_image(fig)

    def scatter_plot(self, x_name: str, y_names: list[str]) -> Image:
        """
        Create a scatter plot for two columns in the table.

        Parameters
        ----------
        x_name:
            The name of the column to be plotted on the x-axis.
        y_names:
            The name(s) of the column(s) to be plotted on the y-axis.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        ColumnNotFoundError
            If a column does not exist.
        TypeError
            If a column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": [2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> image = table.plot.scatter_plot("a", ["b"])
        """
        _plot_validation(self._table, x_name, y_names)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for y_name in y_names:
            ax.scatter(
                x=self._table.get_column(x_name)._series,
                y=self._table.get_column(y_name)._series,
                s=64,  # marker size
                linewidth=1,
                edgecolor="white",
                label=y_name,
            )
        if len(y_names) > 1:
            name = "values"
        else:
            name = y_names[0]
        ax.set(
            xlabel=x_name,
            ylabel=name,
        )
        ax.legend()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        fig.tight_layout()

        return _figure_to_image(fig)

    def moving_average_plot(self, x_name: str, y_name: str, window_size: int) -> Image:
        """
        Create a moving average plot for the y column and plot it by the x column in the table.

        Parameters
        ----------
        x_name:
            The name of the column to be plotted on the x-axis.
        y_name:
            The name of the column to be plotted on the y-axis.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        ColumnNotFoundError
            If a column does not exist.
        TypeError
            If a column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": [2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> image = table.plot.moving_average_plot("a", "b", window_size = 2)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import polars as pl

        _plot_validation(self._table, x_name, [y_name])
        for name in [x_name, y_name]:
            if self._table.get_column(name).missing_value_count() >= 1:
                raise ValueError(
                    f"there are missing values in column '{name}', use transformation to fill missing values "
                    f"or drop the missing values. For a moving average no missing values are allowed.",
                )

        # Calculate the moving average
        mean_col = pl.col(y_name).mean().alias(y_name)
        grouped = self._table._lazy_frame.sort(x_name).group_by(x_name).agg(mean_col).collect()
        data = grouped
        moving_average = data.select([pl.col(y_name).rolling_mean(window_size).alias("moving_average")])
        # set up the arrays for plotting
        y_data_with_nan = moving_average["moving_average"].to_numpy()
        nan_mask = ~np.isnan(y_data_with_nan)
        y_data = y_data_with_nan[nan_mask]
        x_data = data[x_name].to_numpy()[nan_mask]
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, label="moving average")
        ax.set(
            xlabel=x_name,
            ylabel=y_name,
        )
        ax.legend()
        if self._table.get_column(x_name).is_temporal:
            ax.set_xticks(x_data)  # Set x-ticks to the x data points
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        fig.tight_layout()

        return _figure_to_image(fig)


def _plot_validation(table: Table, x_name: str, y_names: list[str]) -> None:
    y_names.append(x_name)
    _check_columns_exist(table, y_names)
    y_names.remove(x_name)
    _check_columns_are_numeric(table, y_names)

    if not table.get_column(x_name).is_numeric and not table.get_column(x_name).is_temporal:
        raise ColumnTypeError(x_name)
