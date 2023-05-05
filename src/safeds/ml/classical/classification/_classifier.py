from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import precision_score as sk_precision_score

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.exceptions import UntaggedTableError


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
        expected = validation_or_test_set.target
        predicted = self.predict(validation_or_test_set.features).target

        return sk_accuracy_score(expected._data, predicted._data)

    def precision(self, validation_or_test_set: TaggedTable, positive_class=1) -> float:
        """
        Compute the classifier's precision on the given data.

        Parameters
        ----------
        validation_or_test_set : TaggedTable
            The validation or test set.
        positive_class : int | str
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        precision : float
            The calculated precision score, i.e. the ratio of correctly predicted positives to all predicted positives.
            Returns 1 if no predictions are made.
        """
        expected = validation_or_test_set.target
        predicted = self.predict(validation_or_test_set.features).target

        if len(expected) != len(predicted):
            raise AssertionError("Different length of 'expected' and 'predicted' vectors.")

        true_positive, false_positive = 0, 0

        for i in range(len(expected)):
            if predicted[i] == positive_class:
                if expected[i] == predicted[i]:
                    true_positive += 1
                else:
                    false_positive += 1

        if (true_positive+false_positive) == 0:
            return 1.0
        return true_positive / (true_positive + false_positive)
