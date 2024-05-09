from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
from safeds.data.tabular.containers import ExperimentalTable, Table
from safeds.exceptions import PlainTableError

if TYPE_CHECKING:
    from typing import Any

    from sklearn.base import ClassifierMixin


class Classifier(ABC):
    """Abstract base class for all classifiers."""

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for a classifier.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.__class__.__qualname__, self.is_fitted)

    @abstractmethod
    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> Classifier:
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
        """

    @abstractmethod
    def predict(
        self,
        dataset: Table | ExperimentalTable | ExperimentalTabularDataset,
    ) -> TabularDataset:
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
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the classifier is fitted."""

    @abstractmethod
    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Classifier.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_metrics(
        self,
        validation_or_test_set: TabularDataset | ExperimentalTabularDataset,
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

    def accuracy(self, validation_or_test_set: TabularDataset | ExperimentalTabularDataset) -> float:
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
        from sklearn.metrics import accuracy_score as sk_accuracy_score

        if not isinstance(validation_or_test_set, TabularDataset) and isinstance(validation_or_test_set, Table):
            raise PlainTableError

        if isinstance(validation_or_test_set, TabularDataset):
            expected_values = validation_or_test_set.target
        else:  # pragma: no cover
            expected_values = validation_or_test_set.target._series
        predicted_values = self.predict(validation_or_test_set.features).target._data

        # TODO: more efficient implementation using polars
        return sk_accuracy_score(expected_values._data, predicted_values)

    def precision(
        self,
        validation_or_test_set: TabularDataset | ExperimentalTabularDataset,
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
        if not isinstance(validation_or_test_set, TabularDataset) and isinstance(validation_or_test_set, Table):
            raise PlainTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        n_true_positives = 0
        n_false_positives = 0

        # TODO: more efficient implementation using polars
        for expected_value, predicted_value in zip(expected_values, predicted_values, strict=True):
            if predicted_value == positive_class:
                if expected_value == positive_class:
                    n_true_positives += 1
                else:
                    n_false_positives += 1

        if (n_true_positives + n_false_positives) == 0:
            return 1.0
        return n_true_positives / (n_true_positives + n_false_positives)

    def recall(self, validation_or_test_set: TabularDataset | ExperimentalTabularDataset, positive_class: Any) -> float:
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
        if not isinstance(validation_or_test_set, TabularDataset) and isinstance(validation_or_test_set, Table):
            raise PlainTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        n_true_positives = 0
        n_false_negatives = 0

        # TODO: more efficient implementation using polars
        for expected_value, predicted_value in zip(expected_values, predicted_values, strict=True):
            if predicted_value == positive_class:
                if expected_value == positive_class:
                    n_true_positives += 1
            elif expected_value == positive_class:
                n_false_negatives += 1

        if (n_true_positives + n_false_negatives) == 0:
            return 1.0
        return n_true_positives / (n_true_positives + n_false_negatives)

    def f1_score(
        self,
        validation_or_test_set: TabularDataset | ExperimentalTabularDataset,
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
        if not isinstance(validation_or_test_set, TabularDataset) and isinstance(validation_or_test_set, Table):
            raise PlainTableError

        expected_values = validation_or_test_set.target
        predicted_values = self.predict(validation_or_test_set.features).target

        n_true_positives = 0
        n_false_negatives = 0
        n_false_positives = 0

        # TODO: more efficient implementation using polars
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
