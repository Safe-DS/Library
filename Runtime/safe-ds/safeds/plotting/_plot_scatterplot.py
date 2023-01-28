import matplotlib.pyplot as plt
import seaborn as sns
from safeds.data import Table
from safeds.exceptions import UnknownColumnNameError


def plot_scatterplot(table: Table, x: str, y: str) -> None:
    """
    Plot two columns against each other in a scatterplot.

    Parameters
    ----------
    table : Table
        The table containing the data to be plotted.
    x : str
        The column name of the column to be plotted on the x-Axis.
    y : str
        The column name of the column to be plotted on the y-Axis.

    Raises
    ---------
    UnknownColumnNameError
        If either of the columns do not exist.
    """
    # noinspection PyProtectedMember
    if not table.has_column(x):
        raise UnknownColumnNameError([x])
    if not table.has_column(y):
        raise UnknownColumnNameError([y])

    ax = sns.scatterplot(
        data=table._data,
        x=table.schema._get_column_index_by_name(x),
        y=table.schema._get_column_index_by_name(y),
    )
    ax.set(xlabel=x, ylabel=y)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, horizontalalignment="right"
    )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
    plt.tight_layout()
    plt.show()
