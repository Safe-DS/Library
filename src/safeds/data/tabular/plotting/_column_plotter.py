from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

    def box_plot(self, *, theme: Literal["dark", "light"] = "light") -> Image:
        """
        Create a box plot for the values in the column. This is only possible for numeric columns.

        Parameters
        ----------
        theme:
            The color theme of the plot. Default is "light".

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

        def _set_boxplot_colors(box: dict, theme: str) -> None:
            if theme == "dark":
                for median in box["medians"]:
                    median.set(color="orange", linewidth=1.5)

                for box_part in box["boxes"]:
                    box_part.set(color="white", linewidth=1.5, facecolor="cyan")

                for whisker in box["whiskers"]:
                    whisker.set(color="white", linewidth=1.5)

                for cap in box["caps"]:
                    cap.set(color="white", linewidth=1.5)

                for flier in box["fliers"]:
                    flier.set(marker="o", color="white", alpha=0.5)
            else:
                for median in box["medians"]:
                    median.set(color="orange", linewidth=1.5)

                for box_part in box["boxes"]:
                    box_part.set(color="black", linewidth=1.5, facecolor="blue")

                for whisker in box["whiskers"]:
                    whisker.set(color="black", linewidth=1.5)

                for cap in box["caps"]:
                    cap.set(color="black", linewidth=1.5)

                for flier in box["fliers"]:
                    flier.set(marker="o", color="black", alpha=0.5)

        style = "dark_background" if theme == "dark" else "default"
        with plt.style.context(style):
            if theme == "dark":
                plt.rcParams.update(
                    {
                        "text.color": "white",
                        "axes.labelcolor": "white",
                        "axes.edgecolor": "white",
                        "xtick.color": "white",
                        "ytick.color": "white",
                        "grid.color": "gray",
                        "grid.linewidth": 0.5,
                    },
                )
            else:
                plt.rcParams.update(
                    {
                        "grid.linewidth": 0.5,
                    },
                )

            fig, ax = plt.subplots()
            box = ax.boxplot(
                self._column._series.drop_nulls(),
                patch_artist=True,
            )

            _set_boxplot_colors(box, theme)

            ax.set(title=self._column.name)
            ax.set_xticks([])
            ax.yaxis.grid(visible=True)
            fig.tight_layout()

            return _figure_to_image(fig)

    def violin_plot(self, *, theme: Literal["dark", "light"] = "light") -> Image:
        """
        Create a violin plot for the values in the column. This is only possible for numeric columns.

        Parameters
        ----------
        theme:
            The color theme of the plot. Default is "light".

        Returns
        -------
        plot:
            The violin plot as an image.

        Raises
        ------
        TypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> violinplot = column.plot.violin_plot()
        """
        if self._column.row_count > 0:
            _check_column_is_numeric(self._column, operation="create a violin plot")
        from math import nan

        import matplotlib.pyplot as plt

        style = "dark_background" if theme == "dark" else "default"
        with plt.style.context(style):
            if theme == "dark":
                plt.rcParams.update(
                    {
                        "text.color": "white",
                        "axes.labelcolor": "white",
                        "axes.edgecolor": "white",
                        "xtick.color": "white",
                        "ytick.color": "white",
                        "grid.color": "gray",
                        "grid.linewidth": 0.5,
                    },
                )
            else:
                plt.rcParams.update(
                    {
                        "grid.linewidth": 0.5,
                    },
                )

            fig, ax = plt.subplots()
            data = self._column._series.drop_nulls()
            if len(data) == 0:
                data = [nan, nan]
            ax.violinplot(
                data,
            )

            ax.set(title=self._column.name)

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
            The color theme of the plot. Default is "light".

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
        import matplotlib.pyplot as plt

        style = "dark_background" if theme == "dark" else "default"
        with plt.style.context(style):
            if theme == "dark":
                plt.rcParams.update(
                    {
                        "text.color": "white",
                        "axes.labelcolor": "white",
                        "axes.edgecolor": "white",
                        "xtick.color": "white",
                        "ytick.color": "white",
                    },
                )

            return self._column.to_table().plot.histograms(max_bin_count=max_bin_count)

    def lag_plot(self, lag: int, *, theme: Literal["dark", "light"] = "light") -> Image:
        """
        Create a lag plot for the values in the column.

        Parameters
        ----------
        lag:
            The amount of lag.
        theme:
            The color theme of the plot. Default is "light".

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
        import matplotlib.pyplot as plt

        style = "dark_background" if theme == "dark" else "default"
        with plt.style.context(style):
            if theme == "dark":
                plt.rcParams.update(
                    {
                        "text.color": "white",
                        "axes.labelcolor": "white",
                        "axes.edgecolor": "white",
                        "xtick.color": "white",
                        "ytick.color": "white",
                    },
                )

            if self._column.row_count > 0:
                _check_column_is_numeric(self._column, operation="create a lag plot")

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
