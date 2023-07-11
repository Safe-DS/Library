from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.neighbors import KNeighborsClassifier as sk_KNeighborsClassifier

from safeds.exceptions import ClosedBound, DatasetMissesDataError, OutOfBoundsError
from safeds.ml.classical._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class KNearestNeighbors(Classifier):
    """
    K-nearest-neighbors classification.

    Parameters
    ----------
    number_of_neighbors : int
        The number of neighbors to use for interpolation. Has to be greater than 0 (validated in the constructor) and
        less than or equal to the sample size (validated when calling `fit`).

    Raises
    ------
    OutOfBoundsError
        If `number_of_neighbors` is less than 1.
    """

    def __init__(self, number_of_neighbors: int) -> None:
        # Validation
        if number_of_neighbors < 1:
            raise OutOfBoundsError(number_of_neighbors, name="number_of_neighbors", lower_bound=ClosedBound(1))

        # Hyperparameters
        self._number_of_neighbors = number_of_neighbors

        # Internal state
        self._wrapped_classifier: sk_KNeighborsClassifier | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    @property
    def number_of_neighbors(self) -> int:
        """
        Get the number of neighbors used for interpolation.

        Returns
        -------
        result: int
            The number of neighbors.
        """
        return self._number_of_neighbors

    def fit(self, training_set: TaggedTable) -> KNearestNeighbors:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : KNearestNeighbors
            The fitted classifier.

        Raises
        ------
        ValueError
            If `number_of_neighbors` is greater than the sample size.
        LearningError
            If the training data contains invalid values or if the training failed.
        UntaggedTableError
            If the table is untagged.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        if training_set.number_of_rows == 0:
            raise DatasetMissesDataError
        if self._number_of_neighbors > training_set.number_of_rows:
            raise ValueError(
                (
                    f"The parameter 'number_of_neighbors' ({self._number_of_neighbors}) has to be less than or equal to"
                    f" the sample size ({training_set.number_of_rows})."
                ),
            )
        wrapped_classifier = self._get_sklearn_classifier()
        fit(wrapped_classifier, training_set)

        result = KNearestNeighbors(self._number_of_neighbors)
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
        NonNumericColumnError
            If the dataset contains non-numerical values.
        MissingValuesColumnError
            If the dataset contains missing values.
        DatasetMissesDataError
            If the dataset contains no rows.
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

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier: ClassifierMixin
            The sklearn Classifier.
        """
        return sk_KNeighborsClassifier(self._number_of_neighbors, n_jobs=-1)
