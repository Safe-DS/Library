from enum import Enum


class RegressorMetric(Enum):
    """An Enum of possible Metrics for a Regressor."""

    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEDIAN_ABSOLUTE_DEVIATION = "median_absolute_deviation"
    COEFFICIENT_OF_DETERMINATION = "coefficient_of_determination"
