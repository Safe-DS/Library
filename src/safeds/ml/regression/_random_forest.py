from __future__ import annotations

from typing import Optional

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml._util_sklearn import fit, predict
from sklearn.ensemble import RandomForestRegressor as sk_RandomForestRegressor

from ._regressor import Regressor


class RandomForest(Regressor):
    """
    This class implements Random Forest regression. It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._wrapped_regressor: Optional[sk_RandomForestRegressor] = None
        self._target_name: Optional[str] = None

    def fit(self, training_set: TaggedTable) -> RandomForest:
        """
        Fit this model given a tagged table.

        Parameters
        ----------
        training_set : TaggedTable
            The tagged table containing the feature and target vectors.

        Raises
        ------
        LearningError
            If the tagged table contains invalid values or if the training failed.
        """
        wrapped_regressor = sk_RandomForestRegressor(n_jobs=-1)
        fit(wrapped_regressor, training_set)

        result = RandomForest()
        result._wrapped_regressor = wrapped_regressor
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
        return predict(self._wrapped_regressor, dataset, self._target_name)
