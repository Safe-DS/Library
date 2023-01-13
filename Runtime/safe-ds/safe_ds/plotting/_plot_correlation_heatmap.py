import matplotlib.pyplot as plt
import seaborn as sns
from safe_ds.data import Table
from safe_ds.exceptions import NonNumericColumnError


def plot_correlation_heatmap(table: Table) -> None:
    """
    Plot a correlation heatmap of an entire table. This function can only plot real numerical data

    Parameters
    ----------
    table : Table
                The column you want to plot

    Raises
    -------
    TypeError
        if the table contains non-numerical data or complex data
    """
    # noinspection PyProtectedMember
    for column in table.to_columns():
        if not column.type.is_numeric():
            raise NonNumericColumnError(column.name)
    # noinspection PyProtectedMember
    sns.heatmap(data=table._data.corr(), vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()
