from __future__ import annotations

from typing import Optional

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml._util_sklearn import fit, predict
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from ._classifier import Classifier


class LogisticRegression(Classifier):
    """
    This class implements regularized logistic regression. It is used as a classifier model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._wrapped_classifier: Optional[sk_LogisticRegression] = None
        self._target_name: Optional[str] = None

    def fit(self, training_set: TaggedTable) -> LogisticRegression:
        """
        Create a new classifier based on this one and fit it with the given training data. This classifier is not
        modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : LogisticRegression
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """

        wrapped_classifier = sk_LogisticRegression(n_jobs=-1)
        fit(wrapped_classifier, training_set)

        result = LogisticRegression()
        result._wrapped_classifier = wrapped_classifier
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
        PredictionError
            If prediction with the given dataset failed.
        """
        return predict(self._wrapped_classifier, dataset, self._target_name)
