from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.ensemble import GradientBoostingClassifier as sk_GradientBoostingClassifier

from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError
from safeds.ml.classical._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class GradientBoosting(Classifier):
    """
    Gradient boosting classification.

    Parameters
    ----------
    number_of_trees: int
        The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large
        number usually results in better performance.
    learning_rate : float
        The larger the value, the more the model is influenced by each additional tree. If the learning rate is too
        low, the model might underfit. If the learning rate is too high, the model might overfit.

    Raises
    ------
    OutOfBoundsError
        If `number_of_trees` or `learning_rate` is less than or equal to 0.
    """

    def __init__(self, *, number_of_trees: int = 100, learning_rate: float = 0.1) -> None:
        # Validation
        if number_of_trees < 1:
            raise OutOfBoundsError(number_of_trees, name="number_of_trees", lower_bound=ClosedBound(1))
        if learning_rate <= 0:
            raise OutOfBoundsError(learning_rate, name="learning_rate", lower_bound=OpenBound(0))

        # Hyperparameters
        self._number_of_trees = number_of_trees
        self._learning_rate = learning_rate

        # Internal state
        self._wrapped_classifier: sk_GradientBoostingClassifier | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    @property
    def number_of_trees(self) -> int:
        """
        Get the number of trees (estimators) in the ensemble.

        Returns
        -------
        result: int
            The number of trees.
        """
        return self._number_of_trees

    @property
    def learning_rate(self) -> float:
        """
        Get the learning rate.

        Returns
        -------
        result: float
            The learning rate.
        """
        return self._learning_rate

    def fit(self, training_set: TaggedTable) -> GradientBoosting:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : GradientBoosting
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        UntaggedTableError
            If the table is untagged.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        wrapped_classifier = self._get_sklearn_classifier()
        fit(wrapped_classifier, training_set)

        result = GradientBoosting(number_of_trees=self._number_of_trees, learning_rate=self._learning_rate)
        result._wrapped_classifier = wrapped_classifier
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
        NonNumericColumnError
            If the dataset contains non-numerical values.
        MissingValuesColumnError
            If the dataset contains missing values.
        DatasetMissesDataError
            If the dataset contains no rows.
        """
        return predict(self._wrapped_classifier, dataset, self._feature_names, self._target_name)

    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the classifier is fitted.
        """
        return self._wrapped_classifier is not None

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier: ClassifierMixin
            The sklearn Classifier.
        """
        return sk_GradientBoostingClassifier(n_estimators=self._number_of_trees, learning_rate=self._learning_rate)
