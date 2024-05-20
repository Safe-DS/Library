from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _figure_to_image
from safeds._validation._check_columns_are_numeric import _check_column_is_numeric

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import Column


class ColumnPlotter:
    """
    A class that contains plotting methods for a column.

    Parameters
    ----------
    column:
        The column to plot.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("test", [1, 2, 3])
    >>> plotter = column.plot
    """

    def __init__(self, column: Column):
        self._column: Column = column

    def box_plot(self) -> Image:
        """
        Create a box plot for the values in the column. This is only possible for numeric columns.

        Returns
        -------
        plot:
            The box plot as an image.

        Raises
        ------
        TypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> boxplot = column.plot.box_plot()
        """
        if self._column.row_count > 0:
            _check_column_is_numeric(self._column, operation="create a box plot")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.boxplot(
            self._column._series.drop_nulls(),
            patch_artist=True,
        )

        ax.set(title=self._column.name)
        ax.set_xticks([])
        ax.yaxis.grid(visible=True)
        fig.tight_layout()

        return _figure_to_image(fig)

    def histogram(self, *, max_bin_count: int = 10) -> Image:
        """
        Create a histogram for the values in the column.

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
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> histogram = column.plot.histogram()
        """
        return self._column.to_table().plot.histograms(max_bin_count=max_bin_count)

    def lag_plot(self, lag: int) -> Image:
        """
        Create a lag plot for the values in the column.

        Parameters
        ----------
        lag:
            The amount of lag.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        TypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("values", [1, 2, 3, 4])
        >>> image = column.plot.lag_plot(2)
        """
        if self._column.row_count > 0:
            _check_column_is_numeric(self._column, operation="create a lag plot")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        series = self._column._series
        ax.scatter(
            x=series.slice(0, max(len(self._column) - lag, 0)),
            y=series.slice(lag),
        )
        ax.set(
            xlabel="y(t)",
            ylabel=f"y(t + {lag})",
        )
        fig.tight_layout()

        return _figure_to_image(fig)
