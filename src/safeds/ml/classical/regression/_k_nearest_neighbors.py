from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
from safeds.data.tabular.containers import ExperimentalTable, Table
from safeds.exceptions import ClosedBound, DatasetMissesDataError, OutOfBoundsError, PlainTableError
from safeds.ml.classical._util_sklearn import fit, predict

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.neighbors import KNeighborsRegressor as sk_KNeighborsRegressor


class KNearestNeighborsRegressor(Regressor):
    """
    K-nearest-neighbors regression.

    Parameters
    ----------
    number_of_neighbors:
        The number of neighbors to use for interpolation. Has to be greater than 0 (validated in the constructor) and
        less than or equal to the sample size (validated when calling `fit`).

    Raises
    ------
    OutOfBoundsError
        If `number_of_neighbors` is less than 1.
    """

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            self._target_name,
            self._feature_names,
            self._number_of_neighbors,
        )

    def __init__(self, number_of_neighbors: int) -> None:
        # Validation
        if number_of_neighbors < 1:
            raise OutOfBoundsError(number_of_neighbors, name="number_of_neighbors", lower_bound=ClosedBound(1))

        # Hyperparameters
        self._number_of_neighbors = number_of_neighbors

        # Internal state
        self._wrapped_regressor: sk_KNeighborsRegressor | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    @property
    def number_of_neighbors(self) -> int:
        """
        Get the number of neighbors used for interpolation.

        Returns
        -------
        result:
            The number of neighbors.
        """
        return self._number_of_neighbors

    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> KNearestNeighborsRegressor:
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
        ValueError
            If `number_of_neighbors` is greater than the sample size.
        LearningError
            If the training data contains invalid values or if the training failed.
        TypeError
            If a table is passed instead of a tabular dataset.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        if not isinstance(training_set, TabularDataset) and isinstance(training_set, Table):
            raise PlainTableError

        if training_set._table.number_of_rows == 0:
            raise DatasetMissesDataError
        if self._number_of_neighbors > training_set._table.number_of_rows:
            raise ValueError(
                (
                    f"The parameter 'number_of_neighbors' ({self._number_of_neighbors}) has to be less than or equal to"
                    f" the sample size ({training_set._table.number_of_rows})."
                ),
            )

        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = KNearestNeighborsRegressor(self._number_of_neighbors)
        result._wrapped_regressor = wrapped_regressor
        result._feature_names = training_set.features.column_names
        result._target_name = training_set.target.name

        return result

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
        NonNumericColumnError
            If the dataset contains non-numerical values.
        MissingValuesColumnError
            If the dataset contains missing values.
        DatasetMissesDataError
            If the dataset contains no rows.
        """
        return predict(self._wrapped_regressor, dataset, self._feature_names, self._target_name)

    @property
    def is_fitted(self) -> bool:
        """Whether the regressor is fitted."""
        return self._wrapped_regressor is not None

    def _get_sklearn_regressor(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_regressor:
            The sklearn Regressor.
        """
        from sklearn.neighbors import KNeighborsRegressor as sk_KNeighborsRegressor

        return sk_KNeighborsRegressor(self._number_of_neighbors, n_jobs=-1)
