from __future__ import annotations

from abc import ABC, abstractmethod

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ColumnLengthMismatchError
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error


class Regressor(ABC):
    """
    Abstract base class for all regressors.
    """

    @abstractmethod
    def fit(self, training_set: TaggedTable) -> Regressor:
        """
        Create a new regressor based on this one and fit it with the given training data. This regressor is not
        modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor : Regressor
            The fitted regressor.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """

    @abstractmethod
    def predict(self, dataset: Table) -> TaggedTable:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset : Table
            The dataset containing the feature vectors.

        Returns
        -------
        table : TaggedTable
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        PredictionError
            If prediction with the given dataset failed.
        """

    # noinspection PyProtectedMember
    def mean_squared_error(self, validation_or_test_set: TaggedTable) -> float:
        """
        Return the mean squared error, calculated from a given known truth and a column to compare.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.

        Returns
        -------
        mean_squared_error : float
            The calculated mean squared error (the average of the distance of each individual row squared).
        """

        expected = validation_or_test_set.target
        predicted = self.predict(validation_or_test_set.features).target

        _check_metrics_preconditions(predicted, expected)
        return sk_mean_squared_error(expected._data, predicted._data)

    # noinspection PyProtectedMember
    def mean_absolute_error(self, validation_or_test_set: TaggedTable) -> float:
        """
        Return the mean absolute error, calculated from a given known truth and a column to compare.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.

        Returns
        -------
        mean_absolute_error : float
            The calculated mean absolute error (the average of the distance of each individual row).
        """

        expected = validation_or_test_set.target
        predicted = self.predict(validation_or_test_set.features).target

        _check_metrics_preconditions(predicted, expected)
        return sk_mean_absolute_error(expected._data, predicted._data)


# noinspection PyProtectedMember
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
