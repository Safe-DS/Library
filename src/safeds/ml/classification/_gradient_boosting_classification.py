from __future__ import annotations

from typing import Optional

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml._util_sklearn import fit, predict
from sklearn.ensemble import GradientBoostingClassifier as sk_GradientBoostingClassifier

from ._classifier import Classifier


class GradientBoosting(Classifier):
    """
    This class implements gradient boosting classification. It is used as a classifier model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._wrapped_classifier: Optional[sk_GradientBoostingClassifier] = None
        self._feature_names: Optional[list[str]] = None
        self._target_name: Optional[str] = None

    def fit(self, training_set: TaggedTable) -> GradientBoosting:
        """
        Create a new classifier based on this one and fit it with the given training data. This classifier is not
        modified.

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
        """

        wrapped_classifier = sk_GradientBoostingClassifier()
        fit(wrapped_classifier, training_set)

        result = GradientBoosting()
        result._wrapped_classifier = wrapped_classifier
        result._feature_names = training_set.features.get_column_names()
        result._target_name = training_set.target.name

        return result

    # noinspection PyProtectedMember
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
        return predict(self._wrapped_classifier, dataset, self._feature_names, self._target_name)
