from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._util_sklearn import fit, predict

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.linear_model import LinearRegression as sk_LinearRegression

    from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
    from safeds.data.tabular.containers import ExperimentalTable, Table


# TODO: rename to linear regressor
class LinearRegressionRegressor(Regressor):
    """Linear regression."""

    def __hash__(self) -> int:
        return _structural_hash(Regressor.__hash__(self), self._target_name, self._feature_names)

    def __init__(self) -> None:
        # Internal state
        self._wrapped_regressor: sk_LinearRegression | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> LinearRegressionRegressor:
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
        TypeError
            If a table is passed instead of a tabular dataset.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = LinearRegressionRegressor()
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
        from sklearn.linear_model import LinearRegression as sk_LinearRegression

        return sk_LinearRegression(n_jobs=-1)
