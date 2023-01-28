from safeds.data.tabular import Column
from safeds.exceptions import ColumnLengthMismatchError
from sklearn.metrics import mean_absolute_error as mean_absolute_error_sklearn
from sklearn.metrics import mean_squared_error as mean_squared_error_sklearn


def mean_squared_error(actual: Column, expected: Column) -> float:
    """
    Return the mean squared error, calculated from a given known truth and a column to compare.

    Parameters
    ----------
    actual : Column
        The column containing estimated values.
    expected : Column
        The column containing ground truth.

    Returns
    -------
    mean_squared_error : float
        The calculated mean squared error (the average of the distance of each individual row squared).
    """
    _check_metrics_preconditions(actual, expected)
    return mean_squared_error_sklearn(expected._data.tolist(), actual._data.tolist())


def mean_absolute_error(actual: Column, expected: Column) -> float:
    """
    Return the mean absolute error, calculated from a given known truth and a column to compare.

    Parameters
    ----------
    actual: Column
        The column containing estimated values.
    expected: Column
        The column containing ground truth.
    Returns
    -------
    mean_absolute_error : float
        The calculated mean absolute error (the average of the distance of each individual row).
    """
    _check_metrics_preconditions(actual, expected)
    return mean_absolute_error_sklearn(expected._data.tolist(), actual._data.tolist())


def _check_metrics_preconditions(actual: Column, expected: Column) -> None:
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
