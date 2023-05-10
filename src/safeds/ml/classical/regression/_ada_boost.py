from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.ensemble import AdaBoostRegressor as sk_AdaBoostRegressor

from safeds.ml.classical._util_sklearn import fit, predict

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class AdaBoost(Regressor):
    """
    Ada Boost regression.

    Parameters
    ----------
    learner: Classifier
        The learner from which the boosted ensemble is built.
    maximum_number_of_learners: int
        The maximum number of learners at which boosting is terminated. In case of perfect fit, the learning procedure
        is stopped early. Has to be greater than 0.
    learning_rate : float
        Weight applied to each regressor at each boosting iteration. A higher learning rate increases the contribution
        of each regressor. Has to be greater than 0.

    Raises
    ------
    ValueError
        If `maximum_number_of_learners` or `learning_rate` are less than or equal to 0
    """

    def __init__(
        self,
        *,
        learner: Regressor | None = None,
        maximum_number_of_learners: int = 50,
        learning_rate: float = 1.0,
    ) -> None:
        # Validation
        if maximum_number_of_learners <= 0:
            raise ValueError("The parameter 'maximum_number_of_learners' has to be grater than 0.")
        if learning_rate <= 0:
            raise ValueError("The parameter 'learning_rate' has to be greater than 0.")

        # Hyperparameters
        self._learner = learner
        self._maximum_number_of_learners = maximum_number_of_learners
        self._learning_rate = learning_rate

        # Internal state
        self._wrapped_regressor: sk_AdaBoostRegressor | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def fit(self, training_set: TaggedTable) -> AdaBoost:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor : AdaBoost
            The fitted regressor.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = AdaBoost(
            learner=self._learner,
            maximum_number_of_learners=self._maximum_number_of_learners,
            learning_rate=self._learning_rate,
        )
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
        learner = self._learner._get_sklearn_regressor() if self._learner is not None else None
        return sk_AdaBoostRegressor(
            estimator=learner,
            n_estimators=self._maximum_number_of_learners,
            learning_rate=self._learning_rate,
        )
