from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _figure_to_image
from safeds.exceptions import NonNumericColumnError

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
        box_plot:
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
        if self._column.number_of_rows > 0 and not self._column.is_numeric:
            # TODO better error message
            raise NonNumericColumnError(f"{self._column.name} is of type {self._column.type}.")

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

    def histogram(self, *, maximum_number_of_bins: int = 10) -> Image:
        """
        Create a histogram for the values in the column.

        Parameters
        ----------
        maximum_number_of_bins:
            The maximum number of bins to use in the histogram. Default is 10.

        Returns
        -------
        histogram:
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> histogram = column.plot.histogram()
        """
        return self._column.to_table().plot.histograms(maximum_number_of_bins=maximum_number_of_bins)

    def lag_plot(self, lag: int) -> Image:
        """
        Create a lag plot for the values in the column.

        Parameters
        ----------
        lag:
            The amount of lag.

        Returns
        -------
        lag_plot:
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
        if self._column.number_of_rows > 0 and not self._column.is_numeric:
            # TODO better error message
            raise NonNumericColumnError("This time series target contains non-numerical columns.")

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
