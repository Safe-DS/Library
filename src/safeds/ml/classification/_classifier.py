from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sklearn.metrics import accuracy_score as sk_accuracy_score

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table, TaggedTable


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
        """
        expected = validation_or_test_set.target
        predicted = self.predict(validation_or_test_set.features).target

        return sk_accuracy_score(expected._data, predicted._data)
