from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestRegressor as sk_RandomForestRegressor

from safeds.ml.classical._util_sklearn import fit, predict

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class RandomForest(Regressor):
    """Random forest regression.

    Parameters
    ----------
    number_of_trees : int
        The number of trees to be used in the random forest. Has to be greater than 0.

    Raises
    ------
    ValueError
        If `number_of_trees` is less than or equal to 0.
    """

    def __init__(self, *, number_of_trees: int = 100) -> None:
        # Validation
        if number_of_trees < 1:
            raise ValueError("The parameter 'number_of_trees' has to be greater than 0.")

        # Hyperparameters
        self._number_of_trees = number_of_trees

        # Internal state
        self._wrapped_regressor: sk_RandomForestRegressor | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    @property
    def number_of_trees(self) -> int:
        return self._number_of_trees

    def fit(self, training_set: TaggedTable) -> RandomForest:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor : RandomForest
            The fitted regressor.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = RandomForest(number_of_trees=self._number_of_trees)
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
        return sk_RandomForestRegressor(self._number_of_trees, n_jobs=-1)
