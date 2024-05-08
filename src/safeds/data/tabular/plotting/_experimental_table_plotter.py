from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import ExperimentalTable


class ExperimentalTablePlotter:
    def __init__(self, table: ExperimentalTable):
        self._table: ExperimentalTable = table

    def box_plots(self) -> Image:
        raise NotImplementedError

    def correlation_heatmap(self) -> Image:
        raise NotImplementedError

    def histograms(self, *, number_of_bins: int = 10) -> Image:
        raise NotImplementedError

    def line_plot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    def scatter_plot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    # TODO: equivalent to Column.plot_compare_columns that takes a list of column names (index_plot)?

    # def plot_correlation_heatmap(self) -> Image:
    #     """
    #     Plot a correlation heatmap for all numerical columns of this `Table`.
    #
    #     Returns
    #     -------
    #     plot:
    #         The plot as an image.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Table
    #     >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
    #     >>> image = table.plot_correlation_heatmap()
    #     """
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     only_numerical = self.remove_columns_with_non_numerical_values()
    #
    #     if self.number_of_rows == 0:
    #         warnings.warn(
    #             "An empty table has been used. A correlation heatmap on an empty table will show nothing.",
    #             stacklevel=2,
    #         )
    #
    #         with warnings.catch_warnings():
    #             warnings.filterwarnings(
    #                 "ignore",
    #                 message=(
    #                     "Attempting to set identical low and high (xlims|ylims) makes transformation singular;"
    #                     " automatically expanding."
    #                 ),
    #             )
    #             fig = plt.figure()
    #             sns.heatmap(
    #                 data=only_numerical._data.corr(),
    #                 vmin=-1,
    #                 vmax=1,
    #                 xticklabels=only_numerical.column_names,
    #                 yticklabels=only_numerical.column_names,
    #                 cmap="vlag",
    #             )
    #             plt.tight_layout()
    #     else:
    #         fig = plt.figure()
    #         sns.heatmap(
    #             data=only_numerical._data.corr(),
    #             vmin=-1,
    #             vmax=1,
    #             xticklabels=only_numerical.column_names,
    #             yticklabels=only_numerical.column_names,
    #             cmap="vlag",
    #         )
    #         plt.tight_layout()
    #
    #     buffer = io.BytesIO()
    #     fig.savefig(buffer, format="png")
    #     plt.close()  # Prevents the figure from being displayed directly
    #     buffer.seek(0)
    #     return Image.from_bytes(buffer.read())
    #
    # def plot_lineplot(self, x_column_name: str, y_column_name: str) -> Image:
    #     """
    #     Plot two columns against each other in a lineplot.
    #
    #     If there are multiple x-values for a y-value, the resulting plot will consist of a line representing the mean
    #     and the lower-transparency area around the line representing the 95% confidence interval.
    #
    #     Parameters
    #     ----------
    #     x_column_name:
    #         The column name of the column to be plotted on the x-Axis.
    #     y_column_name:
    #         The column name of the column to be plotted on the y-Axis.
    #
    #     Returns
    #     -------
    #     plot:
    #         The plot as an image.
    #
    #     Raises
    #     ------
    #     UnknownColumnNameError
    #         If either of the columns do not exist.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Table
    #     >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
    #     >>> image = table.plot_lineplot("temperature", "sales")
    #     """
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     if not self.has_column(x_column_name) or not self.has_column(y_column_name):
    #         similar_columns_x = self._get_similar_columns(x_column_name)
    #         similar_columns_y = self._get_similar_columns(y_column_name)
    #         raise UnknownColumnNameError(
    #             ([x_column_name] if not self.has_column(x_column_name) else [])
    #             + ([y_column_name] if not self.has_column(y_column_name) else []),
    #             (similar_columns_x if not self.has_column(x_column_name) else [])
    #             + (similar_columns_y if not self.has_column(y_column_name) else []),
    #             )
    #
    #     fig = plt.figure()
    #     ax = sns.lineplot(
    #         data=self._data,
    #         x=x_column_name,
    #         y=y_column_name,
    #     )
    #     ax.set(xlabel=x_column_name, ylabel=y_column_name)
    #     ax.set_xticks(ax.get_xticks())
    #     ax.set_xticklabels(
    #         ax.get_xticklabels(),
    #         rotation=45,
    #         horizontalalignment="right",
    #     )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
    #     plt.tight_layout()
    #
    #     buffer = io.BytesIO()
    #     fig.savefig(buffer, format="png")
    #     plt.close()  # Prevents the figure from being displayed directly
    #     buffer.seek(0)
    #     return Image.from_bytes(buffer.read())
    #
    # def plot_scatterplot(self, x_column_name: str, y_column_name: str) -> Image:
    #     """
    #     Plot two columns against each other in a scatterplot.
    #
    #     Parameters
    #     ----------
    #     x_column_name:
    #         The column name of the column to be plotted on the x-Axis.
    #     y_column_name:
    #         The column name of the column to be plotted on the y-Axis.
    #
    #     Returns
    #     -------
    #     plot:
    #         The plot as an image.
    #
    #     Raises
    #     ------
    #     UnknownColumnNameError
    #         If either of the columns do not exist.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Table
    #     >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
    #     >>> image = table.plot_scatterplot("temperature", "sales")
    #     """
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     if not self.has_column(x_column_name) or not self.has_column(y_column_name):
    #         similar_columns_x = self._get_similar_columns(x_column_name)
    #         similar_columns_y = self._get_similar_columns(y_column_name)
    #         raise UnknownColumnNameError(
    #             ([x_column_name] if not self.has_column(x_column_name) else [])
    #             + ([y_column_name] if not self.has_column(y_column_name) else []),
    #             (similar_columns_x if not self.has_column(x_column_name) else [])
    #             + (similar_columns_y if not self.has_column(y_column_name) else []),
    #             )
    #
    #     fig = plt.figure()
    #     ax = sns.scatterplot(
    #         data=self._data,
    #         x=x_column_name,
    #         y=y_column_name,
    #     )
    #     ax.set(xlabel=x_column_name, ylabel=y_column_name)
    #     ax.set_xticks(ax.get_xticks())
    #     ax.set_xticklabels(
    #         ax.get_xticklabels(),
    #         rotation=45,
    #         horizontalalignment="right",
    #     )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
    #     plt.tight_layout()
    #
    #     buffer = io.BytesIO()
    #     fig.savefig(buffer, format="png")
    #     plt.close()  # Prevents the figure from being displayed directly
    #     buffer.seek(0)
    #     return Image.from_bytes(buffer.read())
    #
    # def plot_boxplots(self) -> Image:
    #     """
    #     Plot a boxplot for every numerical column.
    #
    #     Returns
    #     -------
    #     plot:
    #         The plot as an image.
    #
    #     Raises
    #     ------
    #     NonNumericColumnError
    #         If the table contains only non-numerical columns.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Table
    #     >>> table = Table({"a":[1, 2], "b": [3, 42]})
    #     >>> image = table.plot_boxplots()
    #     """
    #     import matplotlib.pyplot as plt
    #     import pandas as pd
    #     import seaborn as sns
    #
    #     numerical_table = self.remove_columns_with_non_numerical_values()
    #     if numerical_table.number_of_columns == 0:
    #         raise NonNumericColumnError("This table contains only non-numerical columns.")
    #     col_wrap = min(numerical_table.number_of_columns, 3)
    #
    #     data = pd.melt(numerical_table._data, value_vars=numerical_table.column_names)
    #     grid = sns.FacetGrid(data, col="variable", col_wrap=col_wrap, sharex=False, sharey=False)
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             "ignore",
    #             message="Using the boxplot function without specifying `order` is likely to produce an incorrect plot.",
    #         )
    #         grid.map(sns.boxplot, "variable", "value")
    #     grid.set_xlabels("")
    #     grid.set_ylabels("")
    #     grid.set_titles("{col_name}")
    #     for axes in grid.axes.flat:
    #         axes.set_xticks([])
    #     plt.tight_layout()
    #     fig = grid.fig
    #
    #     buffer = io.BytesIO()
    #     fig.savefig(buffer, format="png")
    #     plt.close()  # Prevents the figure from being displayed directly
    #     buffer.seek(0)
    #     return Image.from_bytes(buffer.read())
    #
    # def plot_histograms(self, *, number_of_bins: int = 10) -> Image:
    #     """
    #     Plot a histogram for every column.
    #
    #     Parameters
    #     ----------
    #     number_of_bins:
    #         The number of bins to use in the histogram. Default is 10.
    #
    #     Returns
    #     -------
    #     plot:
    #         The plot as an image.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Table
    #     >>> table = Table({"a": [2, 3, 5, 1], "b": [54, 74, 90, 2014]})
    #     >>> image = table.plot_histograms()
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import pandas as pd
    #
    #     n_cols = min(3, self.number_of_columns)
    #     n_rows = 1 + (self.number_of_columns - 1) // n_cols
    #
    #     if n_cols == 1 and n_rows == 1:
    #         fig, axs = plt.subplots(1, 1, tight_layout=True)
    #         one_col = True
    #     else:
    #         fig, axs = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(n_cols * 3, n_rows * 3))
    #         one_col = False
    #
    #     col_names = self.column_names
    #     for col_name, ax in zip(col_names, axs.flatten() if not one_col else [axs], strict=False):
    #         np_col = np.array(self.get_column(col_name))
    #         bins = min(number_of_bins, len(pd.unique(np_col)))
    #
    #         ax.set_title(col_name)
    #         ax.set_xlabel("")
    #         ax.set_ylabel("")
    #
    #         if self.get_column(col_name).type.is_numeric():
    #             np_col = np_col[~np.isnan(np_col)]
    #
    #             if bins < len(pd.unique(np_col)):
    #                 min_val = np.min(np_col)
    #                 max_val = np.max(np_col)
    #                 hist, bin_edges = np.histogram(self.get_column(col_name), bins, range=(min_val, max_val))
    #
    #                 bars = np.array([])
    #                 for i in range(len(hist)):
    #                     bars = np.append(bars, f"{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}")
    #
    #                 ax.bar(bars, hist, edgecolor="black")
    #                 ax.set_xticks(np.arange(len(hist)), bars, rotation=45, horizontalalignment="right")
    #                 continue
    #
    #         np_col = np_col.astype(str)
    #         unique_values = np.unique(np_col)
    #         hist = np.array([np.sum(np_col == value) for value in unique_values])
    #         ax.bar(unique_values, hist, edgecolor="black")
    #         ax.set_xticks(np.arange(len(unique_values)), unique_values, rotation=45, horizontalalignment="right")
    #
    #     for i in range(len(col_names), n_rows * n_cols):
    #         fig.delaxes(axs.flatten()[i])  # Remove empty subplots
    #
    #     buffer = io.BytesIO()
    #     fig.savefig(buffer, format="png")
    #     plt.close()  # Prevents the figure from being displayed directly
    #     buffer.seek(0)
    #     return Image.from_bytes(buffer.read())
