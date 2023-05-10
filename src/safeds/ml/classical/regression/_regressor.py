from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ColumnLengthMismatchError, UntaggedTableError

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class Regressor(ABC):
    """Abstract base class for all regressors."""

    @abstractmethod
    def fit(self, training_set: TaggedTable) -> Regressor:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

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
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetContainsTargetError
            If the dataset contains the target column already.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
        """

    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the regressor is fitted.
        """

    @abstractmethod
    def _get_sklearn_regressor(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_regressor: RegressorMixin
            The sklearn Regressor.
        """

    # noinspection PyProtectedMember
    def mean_squared_error(self, validation_or_test_set: TaggedTable) -> float:
        """
        Compute the mean squared error (MSE) on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.

        Returns
        -------
        mean_squared_error : float
            The calculated mean squared error (the average of the distance of each individual row squared).

        Raises
        ------
        UntaggedTableError
            If the table is untagged.
        """
        if not isinstance(validation_or_test_set, TaggedTable) and isinstance(validation_or_test_set, Table):
            raise UntaggedTableError
        expected = validation_or_test_set.target
        predicted = self.predict(validation_or_test_set.features).target

        _check_metrics_preconditions(predicted, expected)
        return sk_mean_squared_error(expected._data, predicted._data)

    # noinspection PyProtectedMember
    def mean_absolute_error(self, validation_or_test_set: TaggedTable) -> float:
        """
        Compute the mean absolute error (MAE) of the regressor on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.

        Returns
        -------
        mean_absolute_error : float
            The calculated mean absolute error (the average of the distance of each individual row).

        Raises
        ------
        UntaggedTableError
            If the table is untagged.
        """
        if not isinstance(validation_or_test_set, TaggedTable) and isinstance(validation_or_test_set, Table):
            raise UntaggedTableError
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
            "\n".join([f"{column.name}: {column._data.size}" for column in [actual, expected]]),
        )
