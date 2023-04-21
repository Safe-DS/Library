from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestClassifier as sk_RandomForestClassifier

from safeds.ml.classical._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table, TaggedTable


class RandomForest(Classifier):
    """Random forest classification.

    Parameters
    ----------
    number_of_trees : int
        The number of trees to be used in the random forest. Has to be greater than 0.

    Raises
    ------
    ValueError
        If the number of trees is less than 1.
    """

    def __init__(self, number_of_trees: int = 100) -> None:
        if number_of_trees < 1:
            raise ValueError("The number of trees has to be greater than 0.")
        self.number_of_trees = number_of_trees
        self._wrapped_classifier: sk_RandomForestClassifier | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def fit(self, training_set: TaggedTable) -> RandomForest:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : RandomForest
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        wrapped_classifier = sk_RandomForestClassifier(self.number_of_trees, n_jobs=-1)
        fit(wrapped_classifier, training_set)

        result = RandomForest(self.number_of_trees)
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
