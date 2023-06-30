from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sklearn.metrics import accuracy_score as sk_accuracy_score

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import UntaggedTableError

if TYPE_CHECKING:
    from typing import Any

    from sklearn.base import ClassifierMixin


class Classifier(ABC):
    """Abstract base class for all classifiers."""

    @abstractmethod
    def fit(self, training_set: TaggedTable) -> Classifier:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : Classifier
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """

    @abstractmethod
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

    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the classifier is fitted.
        """

    @abstractmethod
    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier: ClassifierMixin
            The sklearn Classifier.
        """

    # noinspection PyProtectedMember
    def accuracy(self, validation_or_test_set: TaggedTable) -> float:
        """
        Compute the accuracy of the classifier on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.

        Returns
        -------
        accuracy : float
            The calculated accuracy score, i.e. the percentage of equal data.

        Raises
        ------
        UntaggedTableError
            If the table is untagged.
        """
        if not isinstance(validation_or_test_set, TaggedTable) and isinstance(validation_or_test_set, Table):
            raise UntaggedTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        return sk_accuracy_score(expected_values._data, predicted_values._data)

    def precision(self, validation_or_test_set: TaggedTable, positive_class: Any) -> float:
        """
        Compute the classifier's precision on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.
        positive_class : Any
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        precision : float
            The calculated precision score, i.e. the ratio of correctly predicted positives to all predicted positives.
            Return 1 if no positive predictions are made.
        """
        if not isinstance(validation_or_test_set, TaggedTable) and isinstance(validation_or_test_set, Table):
            raise UntaggedTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        n_true_positives = 0
        n_false_positives = 0

        for expected_value, predicted_value in zip(expected_values, predicted_values, strict=True):
            if predicted_value == positive_class:
                if expected_value == positive_class:
                    n_true_positives += 1
                else:
                    n_false_positives += 1

        if (n_true_positives + n_false_positives) == 0:
            return 1.0
        return n_true_positives / (n_true_positives + n_false_positives)

    def recall(self, validation_or_test_set: TaggedTable, positive_class: Any) -> float:
        """
        Compute the classifier's recall on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.
        positive_class : Any
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        recall : float
            The calculated recall score, i.e. the ratio of correctly predicted positives to all expected positives.
            Return 1 if there are no positive expectations.
        """
        if not isinstance(validation_or_test_set, TaggedTable) and isinstance(validation_or_test_set, Table):
            raise UntaggedTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        n_true_positives = 0
        n_false_negatives = 0

        for expected_value, predicted_value in zip(expected_values, predicted_values, strict=True):
            if predicted_value == positive_class:
                if expected_value == positive_class:
                    n_true_positives += 1
            elif expected_value == positive_class:
                n_false_negatives += 1

        if (n_true_positives + n_false_negatives) == 0:
            return 1.0
        return n_true_positives / (n_true_positives + n_false_negatives)

    def f1_score(self, validation_or_test_set: TaggedTable, positive_class: Any) -> float:
        """
        Compute the classifier's $F_1$-score on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.
        positive_class : Any
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        f1_score : float
            The calculated $F_1$-score, i.e. the harmonic mean between precision and recall.
            Return 1 if there are no positive expectations and predictions.
        """
        if not isinstance(validation_or_test_set, TaggedTable) and isinstance(validation_or_test_set, Table):
            raise UntaggedTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        n_true_positives = 0
        n_false_negatives = 0
        n_false_positives = 0

        for expected_value, predicted_value in zip(expected_values, predicted_values, strict=True):
            if predicted_value == positive_class:
                if expected_value == positive_class:
                    n_true_positives += 1
                else:
                    n_false_positives += 1
            elif expected_value == positive_class:
                n_false_negatives += 1

        if (2 * n_true_positives + n_false_positives + n_false_negatives) == 0:
            return 1.0
        return 2 * n_true_positives / (2 * n_true_positives + n_false_positives + n_false_negatives)
