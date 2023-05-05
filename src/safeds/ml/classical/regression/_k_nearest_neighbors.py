from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.neighbors import KNeighborsRegressor as sk_KNeighborsRegressor

from safeds.ml.classical._util_sklearn import fit, predict

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class KNearestNeighbors(Regressor):
    """
    K-nearest-neighbors regression.

    Parameters
    ----------
    number_of_neighbors : int
        The number of neighbors to use for interpolation. Has to be greater than 0 (validated in the constructor) and
        less than or equal to the sample size (validated when calling `fit`).

    Raises
    ------
    ValueError
        If `number_of_neighbors` is less than or equal to 0.
    """

    def __init__(self, number_of_neighbors: int) -> None:
        # Validation
        if number_of_neighbors <= 0:
            raise ValueError("The parameter 'number_of_neighbors' has to be greater than 0.")

        # Hyperparameters
        self._number_of_neighbors = number_of_neighbors

        # Internal state
        self._wrapped_regressor: sk_KNeighborsRegressor | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def fit(self, training_set: TaggedTable) -> KNearestNeighbors:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor : KNearestNeighbors
            The fitted regressor.

        Raises
        ------
        ValueError
            If `number_of_neighbors` is greater than the sample size.
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        if self._number_of_neighbors > training_set.number_of_rows:
            raise ValueError(
                (
                    f"The parameter 'number_of_neighbors' ({self._number_of_neighbors}) has to be less than or equal to"
                    f" the sample size ({training_set.number_of_rows})."
                ),
            )

        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = KNearestNeighbors(self._number_of_neighbors)
        result._wrapped_regressor = wrapped_regressor
        result._feature_names = training_set.features.column_names
        result._target_name = training_set.target.name

        return result

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
        return predict(self._wrapped_regressor, dataset, self._feature_names, self._target_name)

    def is_fitted(self) -> bool:
        """
        Check if the regressor is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the regressor is fitted.
        """
        return self._wrapped_regressor is not None

    def _get_sklearn_regressor(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_regressor: RegressorMixin
            The sklearn Regressor.
        """
        return sk_KNeighborsRegressor(self._number_of_neighbors, n_jobs=-1)
