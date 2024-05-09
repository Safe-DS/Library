from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin
    from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

    from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
    from safeds.data.tabular.containers import ExperimentalTable, Table


class LogisticRegressionClassifier(Classifier):
    """Regularized logistic regression."""

    def __hash__(self) -> int:
        return _structural_hash(Classifier.__hash__(self), self._target_name, self._feature_names)

    def __init__(self) -> None:
        # Internal state
        self._wrapped_classifier: sk_LogisticRegression | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> LogisticRegressionClassifier:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier:
            The fitted classifier.

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
        wrapped_classifier = self._get_sklearn_classifier()
        fit(wrapped_classifier, training_set)

        result = LogisticRegressionClassifier()
        result._wrapped_classifier = wrapped_classifier
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
        return predict(self._wrapped_classifier, dataset, self._feature_names, self._target_name)

    @property
    def is_fitted(self) -> bool:
        """Whether the classifier is fitted."""
        return self._wrapped_classifier is not None

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Classifier.
        """
        from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

        return sk_LogisticRegression(n_jobs=-1)
