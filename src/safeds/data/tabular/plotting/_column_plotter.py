from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt

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
    
    def _apply_theme(self, theme: Literal["dark", "light"]) -> None:
        """
        Apply the specified theme to the plot.

        Parameters
        ----------
        theme:
            The theme for the plot, either "dark" or "light".
        """
        if theme == "dark":
            plt.style.use('dark_background')
            plt.rcParams.update({
                'axes.facecolor': 'black',
                'axes.edgecolor': 'white',
                'grid.color': 'white',
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'
            })
        else:
            plt.style.use('default')
            plt.rcParams.update({
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'grid.color': 'black',
                'text.color': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'
            })

    def box_plot(self, theme: Literal["dark", "light"] = "light" ) -> Image:
        """
        Create a box plot for the values in the column. This is only possible for numeric columns.

        Parameter
        ----------
        theme:
            The theme for the plot, either "dark" or "light". Default is "light"
            
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

        self._apply_theme(theme)

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

    def histogram(self, *, max_bin_count: int = 10, theme: Literal["dark", "light"] = "light") -> Image:
        """
        Create a histogram for the values in the column.

        Parameters
        ----------
        max_bin_count:
            The maximum number of bins to use in the histogram. Default is 10.
        theme:
            The theme for the plot, either "dark" or "light". Default is "light"

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
        self._apply_theme(theme)
        return self._column.to_table().plot.histograms(max_bin_count=max_bin_count)

    def lag_plot(self, lag: int, theme: Literal["dark", "light"] = "light") -> Image:
        """
        Create a lag plot for the values in the column.

        Parameters
        ----------
        lag:
            The amount of lag.
        theme:
            The theme for the plot, either "dark" or "light". Default is "light"

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

        self._apply_theme(theme)

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

