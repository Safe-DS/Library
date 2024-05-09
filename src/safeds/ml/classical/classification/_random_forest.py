from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError
from safeds.ml.classical._util_sklearn import fit, predict

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin
    from sklearn.ensemble import RandomForestClassifier as sk_RandomForestClassifier

    from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
    from safeds.data.tabular.containers import ExperimentalTable, Table


class RandomForestClassifier(Classifier):
    """
    Random forest classification.

    Parameters
    ----------
    number_of_trees:
        The number of trees to be used in the random forest. Has to be greater than 0.
    maximum_depth:
        The maximum depth of each tree. If None, the depth is not limited. Has to be greater than 0.
    minimum_number_of_samples_in_leaves:
        The minimum number of samples that must remain in the leaves of each tree. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `number_of_trees` is less than 1.
    OutOfBoundsError
        If `maximum_depth` is less than 1.
    OutOfBoundsError
        If `minimum_number_of_samples_in_leaves` is less than 1.
    """

    def __init__(
        self,
        *,
        number_of_trees: int = 100,
        maximum_depth: int | None = None,
        minimum_number_of_samples_in_leaves: int = 1,
    ) -> None:
        # Validation
        if number_of_trees < 1:
            raise OutOfBoundsError(number_of_trees, name="number_of_trees", lower_bound=ClosedBound(1))
        if maximum_depth is not None and maximum_depth < 1:
            raise OutOfBoundsError(maximum_depth, name="maximum_depth", lower_bound=ClosedBound(1))
        if minimum_number_of_samples_in_leaves < 1:
            raise OutOfBoundsError(
                minimum_number_of_samples_in_leaves,
                name="minimum_number_of_samples_in_leaves",
                lower_bound=ClosedBound(1),
            )

        # Hyperparameters
        self._number_of_trees: int = number_of_trees
        self._maximum_depth: int | None = maximum_depth
        self._minimum_number_of_samples_in_leaves: int = minimum_number_of_samples_in_leaves

        # Internal state
        self._wrapped_classifier: sk_RandomForestClassifier | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            self._feature_names,
            self._target_name,
            self._number_of_trees,
            self._maximum_depth,
            self._minimum_number_of_samples_in_leaves,
        )

    @property
    def number_of_trees(self) -> int:
        """The number of trees used in the random forest."""
        return self._number_of_trees

    @property
    def maximum_depth(self) -> int | None:
        """The maximum depth of each tree."""
        return self._maximum_depth

    @property
    def minimum_number_of_samples_in_leaves(self) -> int:
        """The minimum number of samples that must remain in the leaves of each tree."""
        return self._minimum_number_of_samples_in_leaves

    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> RandomForestClassifier:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier:
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        TypeError
            If a table is passed instead of a tabular dataset.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        wrapped_classifier = self._get_sklearn_classifier()
        fit(wrapped_classifier, training_set)

        result = RandomForestClassifier(
            number_of_trees=self._number_of_trees,
            maximum_depth=self._maximum_depth,
            minimum_number_of_samples_in_leaves=self._minimum_number_of_samples_in_leaves,
        )
        result._wrapped_classifier = wrapped_classifier
        result._feature_names = training_set.features.column_names
        result._target_name = training_set.target.name

        return result

    def predict(self, dataset: Table | ExperimentalTable | ExperimentalTabularDataset) -> TabularDataset:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset:
            The dataset containing the feature vectors.

        Returns
        -------
        table:
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
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

    @property
    def is_fitted(self) -> bool:
        """Whether the classifier is fitted."""
        return self._wrapped_classifier is not None

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Classifier.
        """
        from sklearn.ensemble import RandomForestClassifier as sk_RandomForestClassifier

        return sk_RandomForestClassifier(
            n_estimators=self._number_of_trees,
            max_depth=self._maximum_depth,
            min_samples_leaf=self._minimum_number_of_samples_in_leaves,
            n_jobs=-1,
        )
