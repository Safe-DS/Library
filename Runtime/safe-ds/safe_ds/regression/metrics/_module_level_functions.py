from safe_ds.data import Column
from safe_ds.exceptions import ColumnLengthMismatchError
from sklearn.metrics import mean_squared_error as mean_squared_error_sklearn


def mean_squared_error(actual: Column, expected: Column) -> float:
    """
    Return the mean squared error, calculated from a given known truth and a column to compare.

    Parameters
    ----------
    actual: Column
        Estimated values column
    expected: Column
        Ground truth column

    Returns
    -------
    mean_squared_error: float
        The calculated mean squared error. The average of the distance of each individual row squared.
    """
    if not actual.type.is_numeric():
        raise TypeError(f"Column 'actual' is not numerical but {actual.type}.")
    if not expected.type.is_numeric():
        raise TypeError(f"Column 'expected' is not numerical but {expected.type}.")

    if actual._data.size != expected._data.size:
        raise ColumnLengthMismatchError(
            "\n".join(
                [f"{column.name}: {column._data.size}" for column in [actual, expected]]
            )
        )

    return mean_squared_error_sklearn(expected._data.tolist(), actual._data.tolist())
