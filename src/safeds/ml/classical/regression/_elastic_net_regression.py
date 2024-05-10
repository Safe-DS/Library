from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from warnings import warn

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError
from safeds.ml.classical._util_sklearn import fit, predict

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.linear_model import ElasticNet as sk_ElasticNet

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class ElasticNetRegressor(Regressor):
    """Elastic net regression.

    Parameters
    ----------
    alpha:
        Controls the regularization of the model. The higher the value, the more regularized it becomes.
    lasso_ratio:
        Number between 0 and 1 that controls the ratio between Lasso and Ridge regularization. If 0, only Ridge
        regularization is used. If 1, only Lasso regularization is used.

    Raises
    ------
    OutOfBoundsError
        If `alpha` is negative or `lasso_ratio` is not between 0 and 1.
    """

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            self._target_name,
            self._feature_names,
            self._alpha,
            self._lasso_ratio,
        )

    def __init__(self, *, alpha: float = 1.0, lasso_ratio: float = 0.5) -> None:
        # Validation
        if alpha < 0:
            raise OutOfBoundsError(alpha, name="alpha", lower_bound=ClosedBound(0))
        if alpha == 0:
            warn(
                (
                    "Setting alpha to zero makes this model equivalent to LinearRegression. You should use "
                    "LinearRegression instead for better numerical stability."
                ),
                UserWarning,
                stacklevel=2,
            )
        if lasso_ratio < 0 or lasso_ratio > 1:
            raise OutOfBoundsError(
                lasso_ratio,
                name="lasso_ratio",
                lower_bound=ClosedBound(0),
                upper_bound=ClosedBound(1),
            )
        elif lasso_ratio == 0:
            warnings.warn(
                (
                    "ElasticNetRegression with lasso_ratio = 0 is essentially RidgeRegression."
                    " Use RidgeRegression instead for better numerical stability."
                ),
                stacklevel=2,
            )
        elif lasso_ratio == 1:
            warnings.warn(
                (
                    "ElasticNetRegression with lasso_ratio = 0 is essentially LassoRegression."
                    " Use LassoRegression instead for better numerical stability."
                ),
                stacklevel=2,
            )

        # Hyperparameters
        self._alpha = alpha
        self._lasso_ratio = lasso_ratio

        # Internal state
        self._wrapped_regressor: sk_ElasticNet | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    @property
    def alpha(self) -> float:
        """
        Get the regularization of the model.

        Returns
        -------
        result:
            The regularization of the model.
        """
        return self._alpha

    @property
    def lasso_ratio(self) -> float:
        """
        Get the ratio between Lasso and Ridge regularization.

        Returns
        -------
        result:
            The ratio between Lasso and Ridge regularization.
        """
        return self._lasso_ratio

    def fit(self, training_set: TabularDataset) -> ElasticNetRegressor:
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

        result = ElasticNetRegressor(alpha=self._alpha, lasso_ratio=self._lasso_ratio)
        result._wrapped_regressor = wrapped_regressor
        result._feature_names = training_set.features.column_names
        result._target_name = training_set.target.name

        return result

    def predict(self, dataset: Table | TabularDataset) -> TabularDataset:
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
        from sklearn.linear_model import ElasticNet as sk_ElasticNet

        return sk_ElasticNet(alpha=self._alpha, l1_ratio=self._lasso_ratio)
