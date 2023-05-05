from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.svm import SVC as sk_SVC  # noqa: N811

from safeds.ml.classical._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table, TaggedTable


class SupportVectorMachine(Classifier):
    """
    Support vector machine.

    Parameters
    ----------
    c: float
        The strength of regularization

    Raises
    ------
    ValueError
        If the strength of regularization is less than or equal to 0.
    """

    def __init__(self, c: float = 1.0) -> None:
        # Internal state
        self._wrapped_classifier: sk_SVC | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

        if c <= 0:
            raise ValueError("The strength of regularization must be strictly positive.")
        self._c = c

    def fit(self, training_set: TaggedTable) -> SupportVectorMachine:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : SupportVectorMachine
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        wrapped_classifier = sk_SVC(C=self._c)
        fit(wrapped_classifier, training_set)

        result = SupportVectorMachine(self._c)
        result._wrapped_classifier = wrapped_classifier
        result._feature_names = training_set.features.column_names
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
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetContainsTargetError
            If the dataset contains the target column already.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
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
