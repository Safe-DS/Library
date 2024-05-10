from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.ml.metrics import ClassificationMetrics

if TYPE_CHECKING:
    from typing import Any

    from sklearn.base import ClassifierMixin



class Classifier(ABC):
    """Abstract base class for all classifiers."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__qualname__, self.is_fitted)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the classifier is fitted."""

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return the names of the feature columns.

        Returns
        -------
        feature_names:
            The names of the feature columns.
        """

    @abstractmethod
    def get_target_name(self) -> str:
        """
        Return the name of the target column.

        Returns
        -------
        target_name:
            The name of the target column.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def fit(self, training_set: TabularDataset) -> Classifier:
        """
        Create a copy of this classifier and fit it with the given training data.

        **Note:** This classifier is not modified.

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
        """

    @abstractmethod
    def predict(
        self,
        dataset: Table | TabularDataset,
    ) -> TabularDataset:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be fitted first.

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
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_metrics(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> Table:
        """
        Summarize the classifier's metrics on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        metrics:
            A table containing the classifier's metrics.

        Raises
        ------
        TypeError
            If a table is passed instead of a tabular dataset.
        """
        accuracy = self.accuracy(validation_or_test_set)
        precision = self.precision(validation_or_test_set, positive_class)
        recall = self.recall(validation_or_test_set, positive_class)
        f1_score = self.f1_score(validation_or_test_set, positive_class)

        return Table(
            {
                "metric": ["accuracy", "precision", "recall", "f1_score"],
                "value": [accuracy, precision, recall, f1_score],
            },
        )

    def accuracy(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the accuracy of the classifier on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        accuracy:
            The calculated accuracy score, i.e. the percentage of equal data.

        Raises
        ------
        TypeError
            If a table is passed instead of a tabular dataset.
        """
        if isinstance(validation_or_test_set, TabularDataset):
            validation_or_test_set = validation_or_test_set.to_table()

        features = validation_or_test_set.remove_columns_except(self.get_feature_names())
        prediction = self.predict(features)

        predicted_values = prediction.target
        expected_values = validation_or_test_set.get_column(self.get_target_name())

        return ClassificationMetrics.accuracy(predicted_values, expected_values)

    def precision(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> float:
        """
        Compute the classifier's precision on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        precision:
            The calculated precision score, i.e. the ratio of correctly predicted positives to all predicted positives.
            Return 1 if no positive predictions are made.
        """
        if isinstance(validation_or_test_set, TabularDataset):
            validation_or_test_set = validation_or_test_set.to_table()

        features = validation_or_test_set.remove_columns_except(self.get_feature_names())
        prediction = self.predict(features)

        predicted_values = prediction.target
        expected_values = validation_or_test_set.get_column(self.get_target_name())

        return ClassificationMetrics.precision(predicted_values, expected_values, positive_class)

    def recall(self, validation_or_test_set: Table | TabularDataset, positive_class: Any) -> float:
        """
        Compute the classifier's recall on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        recall:
            The calculated recall score, i.e. the ratio of correctly predicted positives to all expected positives.
            Return 1 if there are no positive expectations.
        """
        if isinstance(validation_or_test_set, TabularDataset):
            validation_or_test_set = validation_or_test_set.to_table()

        features = validation_or_test_set.remove_columns_except(self.get_feature_names())
        prediction = self.predict(features)

        predicted_values = prediction.target
        expected_values = validation_or_test_set.get_column(self.get_target_name())

        return ClassificationMetrics.recall(predicted_values, expected_values, positive_class)

    def f1_score(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> float:
        """
        Compute the classifier's $F_1$-score on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        f1_score:
            The calculated $F_1$-score, i.e. the harmonic mean between precision and recall.
            Return 1 if there are no positive expectations and predictions.
        """
        if isinstance(validation_or_test_set, TabularDataset):
            validation_or_test_set = validation_or_test_set.to_table()

        features = validation_or_test_set.remove_columns_except(self.get_feature_names())
        prediction = self.predict(features)

        predicted_values = prediction.target
        expected_values = validation_or_test_set.get_column(self.get_target_name())

        return ClassificationMetrics.f1_score(predicted_values, expected_values, positive_class)

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Classifier.
        """
