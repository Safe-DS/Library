from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.ensemble import GradientBoostingClassifier as sk_GradientBoostingClassifier

from safeds.ml._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table, TaggedTable


class GradientBoosting(Classifier):
    """This class implements gradient boosting classification."""

    def __init__(self) -> None:
        self._wrapped_classifier: sk_GradientBoostingClassifier | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def fit(self, training_set: TaggedTable) -> GradientBoosting:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

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

    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the classifier is fitted.
        """
        return self._wrapped_classifier is not None
