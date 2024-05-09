from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from safeds._utils import _figure_to_image
from safeds.exceptions import NonNumericColumnError, UnknownColumnNameError

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import ExperimentalTable


class ExperimentalTablePlotter:
    def __init__(self, table: ExperimentalTable):
        self._table: ExperimentalTable = table

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
        >>> image = table.plot_boxplots()
        """
        # TOOD: implement using matplotlib and polars
        import matplotlib.pyplot as plt
        import seaborn as sns

        numerical_table = self._table.remove_non_numeric_columns()
        if numerical_table.number_of_columns == 0:
            raise NonNumericColumnError("This table contains only non-numerical columns.")
        col_wrap = min(numerical_table.number_of_columns, 3)

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
        >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
        >>> image = table.plot_correlation_heatmap()
        """
        # TODO: implement using matplotlib and polars
        #  https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns

        only_numerical = self._table.remove_non_numeric_columns()

        if self._table.number_of_rows == 0:
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
                fig = plt.figure()
                sns.heatmap(
                    data=only_numerical._data_frame.corr(),
                    vmin=-1,
                    vmax=1,
                    xticklabels=only_numerical.column_names,
                    yticklabels=only_numerical.column_names,
                    cmap="vlag",
                )
                plt.tight_layout()
        else:
            fig = plt.figure()
            sns.heatmap(
                data=only_numerical._data_frame.corr(),
                vmin=-1,
                vmax=1,
                xticklabels=only_numerical.column_names,
                yticklabels=only_numerical.column_names,
                cmap="vlag",
            )
            plt.tight_layout()

        return _figure_to_image(fig)

    def histograms(self, *, number_of_bins: int = 10) -> Image:
        """
        Plot a histogram for every column.

        Parameters
        ----------
        number_of_bins:
            The number of bins to use in the histogram. Default is 10.

        Returns
        -------
        plot:
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [2, 3, 5, 1], "b": [54, 74, 90, 2014]})
        >>> image = table.plot_histograms()
        """
        # TODO: implement using polars
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        n_cols = min(3, self._table.number_of_columns)
        n_rows = 1 + (self._table.number_of_columns - 1) // n_cols

        if n_cols == 1 and n_rows == 1:
            fig, axs = plt.subplots(1, 1, tight_layout=True)
            one_col = True
        else:
            fig, axs = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(n_cols * 3, n_rows * 3))
            one_col = False

        col_names = self._table.column_names
        for col_name, ax in zip(col_names, axs.flatten() if not one_col else [axs], strict=False):
            np_col = np.array(self._table.get_column(col_name))
            bins = min(number_of_bins, len(pd.unique(np_col)))

            ax.set_title(col_name)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if self._table.get_column(col_name).type.is_numeric:
                np_col = np_col[~np.isnan(np_col)]

                if bins < len(pd.unique(np_col)):
                    min_val = np.min(np_col)
                    max_val = np.max(np_col)
                    hist, bin_edges = np.histogram(self._table.get_column(col_name), bins, range=(min_val, max_val))

                    bars = np.array([])
                    for i in range(len(hist)):
                        bars = np.append(bars, f"{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}")

                    ax.bar(bars, hist, edgecolor="black")
                    ax.set_xticks(np.arange(len(hist)), bars, rotation=45, horizontalalignment="right")
                    continue

            np_col = np_col.astype(str)
            unique_values = np.unique(np_col)
            hist = np.array([np.sum(np_col == value) for value in unique_values])
            ax.bar(unique_values, hist, edgecolor="black")
            ax.set_xticks(np.arange(len(unique_values)), unique_values, rotation=45, horizontalalignment="right")

        for i in range(len(col_names), n_rows * n_cols):
            fig.delaxes(axs.flatten()[i])  # Remove empty subplots

        return _figure_to_image(fig)

    def line_plot(self, x_name: str, y_name: str) -> Image:
        """
        Create a line plot for two columns in the table.

        Parameters
        ----------
        x_name:
            The name of the column to be plotted on the x-axis.
        y_name:
            The name of the column to be plotted on the y-axis.

        Returns
        -------
        line_plot:
            The plot as an image.

        Raises
        ------
        KeyError
            If a column does not exist.
        TypeError
            If a column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": [2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> image = table.plot.line_plot("a", "b")
        """
        # TODO: extract validation
        missing_columns = []
        if not self._table.has_column(x_name):
            missing_columns.append(x_name)
        if not self._table.has_column(y_name):
            missing_columns.append(y_name)
        if missing_columns:
            raise UnknownColumnNameError(missing_columns)

        # TODO: pass list of columns names
        if not self._table.get_column(x_name).is_numeric:
            raise NonNumericColumnError(x_name)
        if not self._table.get_column(y_name).is_numeric:
            raise NonNumericColumnError(y_name)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(
            self._table.get_column(x_name)._series,
            self._table.get_column(y_name)._series,
        )
        ax.set(
            xlabel=x_name,
            ylabel=y_name,
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        fig.tight_layout()

        return _figure_to_image(fig)

    def scatter_plot(self, x_name: str, y_name: str) -> Image:
        """
        Create a scatter plot for two columns in the table.

        Parameters
        ----------
        x_name:
            The name of the column to be plotted on the x-axis.
        y_name:
            The name of the column to be plotted on the y-axis.

        Returns
        -------
        scatter_plot:
            The plot as an image.

        Raises
        ------
        KeyError
            If a column does not exist.
        TypeError
            If a column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": [2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> image = table.plot.scatter_plot("a", "b")
        """
        # TODO: merge with line_plot?
        # TODO: extract validation
        missing_columns = []
        if not self._table.has_column(x_name):
            missing_columns.append(x_name)
        if not self._table.has_column(y_name):
            missing_columns.append(y_name)
        if missing_columns:
            raise UnknownColumnNameError(missing_columns)

        # TODO: pass list of columns names
        if not self._table.get_column(x_name).is_numeric:
            raise NonNumericColumnError(x_name)
        if not self._table.get_column(y_name).is_numeric:
            raise NonNumericColumnError(y_name)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(
            x=self._table.get_column(x_name)._series,
            y=self._table.get_column(y_name)._series,
        )
        ax.set(
            xlabel=x_name,
            ylabel=y_name,
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        fig.tight_layout()

        return _figure_to_image(fig)

    # TODO: equivalent to Column.plot_compare_columns that takes a list of column names (index_plot)?
