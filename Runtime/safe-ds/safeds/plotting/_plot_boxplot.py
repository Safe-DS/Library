import matplotlib.pyplot as plt
import seaborn as sns
from safeds.data import Column
from safeds.exceptions import NonNumericColumnError


def plot_boxplot(column: Column) -> None:
    """
    Plot a column in a boxplot. This function can only plot real numerical data.

    Parameters
    ----------
    column : Column
        The column to be plotted.

    Raises
    -------
    TypeError
        If the column contains non-numerical data or complex data.
    """
    # noinspection PyProtectedMember
    for data in column._data:
        if (
            not isinstance(data, int)
            and not isinstance(data, float)
            and not isinstance(data, complex)
        ):
            raise NonNumericColumnError(column.name)
        if isinstance(data, complex):
            raise TypeError(
                "The column contains complex data. Boxplots cannot plot the imaginary part of complex "
                "data. Please provide a Column with only real numbers"
            )
    # noinspection PyProtectedMember
    ax = sns.boxplot(data=column._data)
    ax.set(xlabel=column.name)
    plt.tight_layout()
    plt.show()
