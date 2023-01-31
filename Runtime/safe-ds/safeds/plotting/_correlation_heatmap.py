import matplotlib.pyplot as plt
import seaborn as sns
from safeds.data.tabular import Table
from safeds.exceptions import NonNumericColumnError


def correlation_heatmap(table: Table) -> None:
    """
    Plot a correlation heatmap of an entire table. This function can only plot real numerical data.

    Parameters
    ----------
    table : Table
        The column to be plotted.

    Raises
    -------
    TypeError
        If the table contains non-numerical data or complex data.
    """
    # noinspection PyProtectedMember
    for column in table.to_columns():
        if not column.type.is_numeric():
            raise NonNumericColumnError(column.name)
    # noinspection PyProtectedMember
    sns.heatmap(
        data=table._data.corr(),
        vmin=-1,
        vmax=1,
        xticklabels=table.get_column_names(),
        yticklabels=table.get_column_names(),
        cmap="vlag",
    )
    plt.tight_layout()
    plt.show()
