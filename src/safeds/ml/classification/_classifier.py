from __future__ import annotations

from abc import ABC, abstractmethod

from safeds.data.tabular.containers import Table, TaggedTable
from sklearn.metrics import accuracy_score as sk_accuracy_score


class Classifier(ABC):
    """
    Abstract base class for all classifiers.
    """

    @abstractmethod
    def fit(self, training_set: TaggedTable) -> Classifier:
        """
        Create a new classifier based on this one and fit it with the given training data. This classifier is not
        modified.

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
        PredictionError
            If prediction with the given dataset failed.
        """

    # noinspection PyProtectedMember
    def accuracy(self, validation_or_test_set: TaggedTable) -> float:
        """
        Predicts the target values for the features in the validation or test set and compares it to the expected
        results.

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
