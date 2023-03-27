from __future__ import annotations

from typing import Optional

from sklearn.ensemble import AdaBoostClassifier as sk_AdaBoostClassifier

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml._util_sklearn import fit, predict
from ._classifier import Classifier


class AdaBoost(Classifier):
    """
    This class implements Ada Boost classification. It is used as a classifier model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._wrapped_classifier: Optional[sk_AdaBoostClassifier] = None
        self._target_name: Optional[str] = None

    def fit(self, training_set: TaggedTable) -> AdaBoost:
        """
        Fit this model given a tagged table.

        Parameters
        ----------
        training_set : TaggedTable
            The tagged table containing the feature and target vectors.

        Raises
        ------
        LearningError
            If the tagged table contains invalid values or if the training failed.
        """

        wrapped_classifier = sk_AdaBoostClassifier()
        fit(wrapped_classifier, training_set)

        result = AdaBoost()
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
