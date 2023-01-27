from safe_ds.data import Column
from sklearn.metrics import accuracy_score


def accuracy(actual: Column, expected: Column) -> float:
    """
    Compares the expected column and the predicted column and returns the accuracy.

    Parameters
    ----------
    actual : Column
        The estimated values column.
    expected : Column
        The expected values column.

    Returns
    -------
    accuracy : float
        The calculated accuracy score. The percentage of equal data.
    """
    return accuracy_score(actual._data, expected._data)
