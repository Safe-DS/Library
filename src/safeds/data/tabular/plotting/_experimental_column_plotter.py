from __future__ import annotations

import io
from typing import TYPE_CHECKING

from safeds.data.image.containers import Image
from safeds.exceptions import NonNumericColumnError

if TYPE_CHECKING:
    from safeds.data.tabular.containers import ExperimentalColumn


class ExperimentalColumnPlotter:
    def __init__(self, column: ExperimentalColumn):
        self._column: ExperimentalColumn = column

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
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> boxplot = column.plot.box_plot()
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self._column.is_numeric:
            raise NonNumericColumnError(f"{self._column.name} is of type {self._column.type}.")

        fig = plt.figure()
        ax = sns.boxplot(data=self._column)
        ax.set(title=self._column.name)
        ax.set_xticks([])
        ax.set_ylabel("")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def histogram(self, *, number_of_bins: int = 10) -> Image:
        """
        Create a histogram for the values in the column.

        Parameters
        ----------
        number_of_bins:
            The number of bins to use in the histogram. Default is 10.

        Returns
        -------
        histogram:
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> histogram = column.plot.histogram()
        """
        return self._column.to_table().plot.histograms(number_of_bins=number_of_bins)

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
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> table = ExperimentalColumn("values", [1,2,3,4,3,2])
        >>> image = table.plot.lag_plot(2)
        """
        import matplotlib.pyplot as plt

        if not self._column.is_numeric:
            raise NonNumericColumnError("This time series target contains non-numerical columns.")

        x_column_name = "y(t)"
        y_column_name = f"y(t + {lag})"

        fig, ax = plt.subplots()
        ax.scatter(x=self._column[:-lag], y=self._column[lag:])
        ax.set(xlabel=x_column_name, ylabel=y_column_name)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())
