import matplotlib.pyplot as plt
import seaborn as sns
from safe_ds.data import Column


def plot_histogram(column: Column) -> None:
    """
    Plot a column in a histogram.

    Parameters
    ----------
    column : Column
        The column you want to plot
    """
    # noinspection PyProtectedMember
    ax = sns.histplot(data=column._data)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, horizontalalignment="right"
    )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
    plt.tight_layout()
    plt.show()
