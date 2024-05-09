from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
from safeds.data.tabular.containers import Column, ExperimentalColumn, ExperimentalTable, Table
from safeds.exceptions import ColumnLengthMismatchError, PlainTableError

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class Regressor(ABC):
    """Abstract base class for all regressors."""

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for a regressor.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.__class__.__qualname__, self.is_fitted)

    @abstractmethod
    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> Regressor:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor:
            The fitted regressor.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """

    @abstractmethod
    def predict(self, dataset: Table | ExperimentalTable | ExperimentalTabularDataset) -> TabularDataset:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset:
            The dataset containing the feature vectors.

        Returns
        -------
        table:
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the regressor is fitted."""

    @abstractmethod
    def _get_sklearn_regressor(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_regressor:
            The sklearn Regressor.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_metrics(self, validation_or_test_set: TabularDataset | ExperimentalTabularDataset) -> Table:
        """
        Summarize the regressor's metrics on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        metrics:
            A table containing the regressor's metrics.

        Raises
        ------
        TypeError
            If a table is passed instead of a tabular dataset.
        """
        mean_absolute_error = self.mean_absolute_error(validation_or_test_set)
        mean_squared_error = self.mean_squared_error(validation_or_test_set)

        return Table(
            {
                "metric": ["mean_absolute_error", "mean_squared_error"],
                "value": [mean_absolute_error, mean_squared_error],
            },
        )

    def mean_absolute_error(self, validation_or_test_set: TabularDataset | ExperimentalTabularDataset) -> float:
        """
        Compute the mean absolute error (MAE) of the regressor on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        mean_absolute_error:
            The calculated mean absolute error (the average of the distance of each individual row).

        Raises
        ------
        TypeError
            If a table is passed instead of a tabular dataset.
        """
        from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error

        if not isinstance(validation_or_test_set, TabularDataset) and isinstance(validation_or_test_set, Table):
            raise PlainTableError

        if isinstance(validation_or_test_set, TabularDataset):
            expected = validation_or_test_set.target
            predicted = self.predict(validation_or_test_set.features).target

            # TODO: more efficient implementation using polars
            _check_metrics_preconditions(predicted, expected)
            return sk_mean_absolute_error(expected._data, predicted._data)
        elif isinstance(validation_or_test_set, ExperimentalTabularDataset):  # pragma: no cover
            expected_2 = validation_or_test_set.target
            predicted_2 = self.predict(validation_or_test_set.features).target

            # TODO: more efficient implementation using polars
            _check_metrics_preconditions_experimental(predicted_2, expected_2)
            return sk_mean_absolute_error(expected_2._series, predicted_2._data)

    # noinspection PyProtectedMember
    def mean_squared_error(self, validation_or_test_set: TabularDataset | ExperimentalTabularDataset) -> float:
        """
        Compute the mean squared error (MSE) on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        mean_squared_error:
            The calculated mean squared error (the average of the distance of each individual row squared).

        Raises
        ------
        TypeError
            If a table is passed instead of a tabular dataset.
        """
        from sklearn.metrics import mean_squared_error as sk_mean_squared_error

        if not isinstance(validation_or_test_set, TabularDataset) and isinstance(validation_or_test_set, Table):
            raise PlainTableError

        if isinstance(validation_or_test_set, TabularDataset):
            expected = validation_or_test_set.target
            predicted = self.predict(validation_or_test_set.features).target

            # TODO: more efficient implementation using polars
            _check_metrics_preconditions(predicted, expected)
            return sk_mean_squared_error(expected._data, predicted._data)
        elif isinstance(validation_or_test_set, ExperimentalTabularDataset):  # pragma: no cover
            expected_2 = validation_or_test_set.target
            predicted_2 = self.predict(validation_or_test_set.features).target

            # TODO: more efficient implementation using polars
            _check_metrics_preconditions_experimental(predicted_2, expected_2)
            return sk_mean_squared_error(expected_2._series, predicted_2._data)


def _check_metrics_preconditions(actual: Column, expected: Column) -> None:
    if not actual.type.is_numeric():
        raise TypeError(f"Column 'actual' is not numerical but {actual.type}.")
    if not expected.type.is_numeric():
        raise TypeError(f"Column 'expected' is not numerical but {expected.type}.")

    if actual._data.size != expected._data.size:
        raise ColumnLengthMismatchError(
            "\n".join([f"{column.name}: {column._data.size}" for column in [actual, expected]]),
        )


def _check_metrics_preconditions_experimental(actual: Column, expected: ExperimentalColumn) -> None:  # pragma: no cover
    if not actual.type.is_numeric():
        raise TypeError(f"Column 'actual' is not numerical but {actual.type}.")
    if not expected.type.is_numeric:
        raise TypeError(f"Column 'expected' is not numerical but {expected.type}.")

    if actual.number_of_rows != expected.number_of_rows:
        raise ColumnLengthMismatchError(
            "\n".join(
                [
                    f"{actual.name}: {actual.number_of_rows}",
                    f"{expected.name}: {expected.number_of_rows}",
                ],
            ),
        )
