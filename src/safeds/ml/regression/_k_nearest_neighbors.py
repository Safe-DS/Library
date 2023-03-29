from __future__ import annotations

from typing import Optional

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml._util_sklearn import fit, predict
from sklearn.neighbors import KNeighborsRegressor as sk_KNeighborsRegressor

from ._regressor import Regressor


class KNearestNeighbors(Regressor):
    """
    This class implements K-nearest-neighbors regressor. It can only be trained on a tagged table.

    Parameters
    ----------
    n_neighbors : int
        The number of neighbors to be interpolated with. Has to be less than or equal than the sample size.
    """

    def __init__(self, n_neighbors: int) -> None:
        self._n_neighbors = n_neighbors

        self._wrapped_regressor: Optional[sk_KNeighborsRegressor] = None
        self._feature_names: Optional[list[str]] = None
        self._target_name: Optional[str] = None

    def fit(self, training_set: TaggedTable) -> KNearestNeighbors:
        """
        Create a new regressor based on this one and fit it with the given training data. This regressor is not
        modified.

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
        LearningError
            If the training data contains invalid values or if the training failed.
        """

        wrapped_regressor = sk_KNeighborsRegressor(self._n_neighbors, n_jobs=-1)
        fit(wrapped_regressor, training_set)

        result = KNearestNeighbors(self._n_neighbors)
        result._wrapped_regressor = wrapped_regressor
        result._feature_names = training_set.features.get_column_names()
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
        PredictionError
            If prediction with the given dataset failed.
        """
        return predict(self._wrapped_regressor, dataset, self._feature_names, self._target_name)
