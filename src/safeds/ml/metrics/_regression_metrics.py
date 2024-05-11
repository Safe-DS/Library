from __future__ import annotations

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ColumnLengthMismatchError


class RegressionMetrics:
    """A collection of regression metrics."""

    @staticmethod
    def summarize(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> Table:
        """
        Summarize regression metrics on the given data.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the regressor.
        expected:
            The expected target values.

        Returns
        -------
        metrics:
            A table containing the regression metrics.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        coefficient_of_determination = RegressionMetrics.coefficient_of_determination(expected, predicted)
        mean_absolute_error = RegressionMetrics.mean_absolute_error(expected, predicted)
        mean_squared_error = RegressionMetrics.mean_squared_error(expected, predicted)
        median_absolute_deviation = RegressionMetrics.median_absolute_deviation(expected, predicted)

        return Table(
            {
                "metric": [
                    "coefficient_of_determination",
                    "mean_absolute_error",
                    "mean_squared_error",
                    "median_absolute_deviation",
                ],
                "value": [
                    coefficient_of_determination,
                    mean_absolute_error,
                    mean_squared_error,
                    median_absolute_deviation,
                ],
            },
        )

    @staticmethod
    def coefficient_of_determination(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> float:
        """
        Compute the coefficient of determination (R²) on the given data.

        The coefficient of determination compares the regressor's predictions to another model that always predicts the
        mean of the target values. It is a measure of how well the regressor explains the variance in the target values.

        The **higher** the coefficient of determination, the better the regressor. Results range from negative infinity
        to 1.0. You can interpret the coefficient of determination as follows:

        | R²         | Interpretation                                                                             |
        | ---------- | ------------------------------------------------------------------------------------------ |
        | 1.0        | The model perfectly predicts the target values. Did you overfit?                           |
        | (0.0, 1.0) | The model is better than predicting the mean of the target values. You should be here.     |
        | 0.0        | The model is as good as predicting the mean of the target values. Try something else.      |
        | (-∞, 0.0)  | The model is worse than predicting the mean of the target values. Something is very wrong. |

        **Note:** Some other libraries call this metric `r2_score`.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the regressor.
        expected:
            The expected target values.

        Returns
        -------
        coefficient_of_determination:
            The calculated coefficient of determination.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        residual_sum_of_squares = (expected._series - predicted._series).pow(2).sum()
        total_sum_of_squares = (expected._series - expected._series.mean()).pow(2).sum()

        if total_sum_of_squares == 0:
            if residual_sum_of_squares == 0:
                return 1.0  # Everything was predicted correctly
            else:
                return 0.0  # Model could not even predict constant data

        return 1 - residual_sum_of_squares / total_sum_of_squares

    @staticmethod
    def mean_absolute_error(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> float:
        """
        Compute the mean absolute error (MAE) on the given data.

        The mean absolute error is the average of the absolute differences between the predicted and expected target
        values. The **lower** the mean absolute error, the better the regressor. Results range from 0.0 to positive
        infinity.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the regressor.
        expected:
            The expected target values.

        Returns
        -------
        mean_absolute_error:
            The calculated mean absolute error.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        if expected.number_of_rows == 0:
            return 0.0  # Everything was predicted correctly (since there is nothing to predict)

        return (expected._series - predicted._series).abs().mean()

    @staticmethod
    def mean_directional_accuracy(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> float:
        """
        Compute the mean directional accuracy (MDA) on the given data.

        This metric compares two consecutive target values and checks if the predicted direction (down/unchanged/up)
        matches the expected direction. The mean directional accuracy is the proportion of correctly predicted
        directions. The **higher** the mean directional accuracy, the better the regressor. Results range from 0.0 to
        1.0.

        This metric is useful for time series data, where the order of the target values has a meaning. It is not useful
        for other types of data. Because of this, it is not included in the `summarize` method.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the regressor.
        expected:
            The expected target values.

        Returns
        -------
        mean_directional_accuracy:
            The calculated mean directional accuracy.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        if expected.number_of_rows == 0:
            return 1.0

        # Calculate the differences between the target values
        predicted_directions = predicted._series.diff().sign()
        expected_directions = expected._series.diff().sign()

        return predicted_directions.eq(expected_directions).mean()

    @staticmethod
    def mean_squared_error(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> float:
        """
        Compute the mean squared error (MSE) on the given data.

        The mean squared error is the average of the squared differences between the predicted and expected target
        values. The **lower** the mean squared error, the better the regressor. Results range from 0.0 to positive
        infinity.

        **Note:** To get the root mean squared error (RMSE), take the square root of the result.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the regressor.
        expected:
            The expected target values.

        Returns
        -------
        mean_squared_error:
            The calculated mean squared error.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        if expected.number_of_rows == 0:
            return 0.0  # Everything was predicted correctly (since there is nothing to predict)

        return (expected._series - predicted._series).pow(2).mean()

    @staticmethod
    def median_absolute_deviation(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> float:
        """
        Compute the median absolute deviation (MAD) on the given data.

        The median absolute deviation is the median of the absolute differences between the predicted and expected
        target values. The **lower** the median absolute deviation, the better the regressor. Results range from 0.0 to
        positive infinity.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the regressor.
        expected:
            The expected target values.

        Returns
        -------
        median_absolute_deviation:
            The calculated median absolute deviation.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        if expected.number_of_rows == 0:
            return 0.0

        return (expected._series - predicted._series).abs().median()


def _extract_target(column_or_dataset: Column | TabularDataset) -> Column:
    """Extract the target column from the given column or dataset."""
    if isinstance(column_or_dataset, TabularDataset):
        return column_or_dataset.target
    else:
        return column_or_dataset


# TODO: collect validation in one place?
def _check_equal_length(column1: Column, column2: Column) -> None:
    """
    Check if the columns have the same length and raise an error if they do not.

    Parameters
    ----------
    column1:
        The first column.
    column2:
        The second column.

    Raises
    ------
    ValueError
        If the columns have different lengths.
    """
    if column1.number_of_rows != column2.number_of_rows:
        ColumnLengthMismatchError("")  # TODO: pass list of columns to exception, let it handle the formatting
