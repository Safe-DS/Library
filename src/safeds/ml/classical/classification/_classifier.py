from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import ModelNotFittedError
from safeds.ml.metrics import ClassificationMetrics

if TYPE_CHECKING:
    from typing import Any

    from sklearn.base import ClassifierMixin

    from safeds.data.tabular.containers import Table


class Classifier(ABC):
    """Abstract base class for all classifiers."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__qualname__, self.is_fitted)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the classifier is fitted."""

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return the names of the feature columns.

        Returns
        -------
        feature_names:
            The names of the feature columns.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """

    @abstractmethod
    def get_target_name(self) -> str:
        """
        Return the name of the target column.

        Returns
        -------
        target_name:
            The name of the target column.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def fit(self, training_set: TabularDataset) -> Classifier:
        """
        Create a copy of this classifier and fit it with the given training data.

        **Note:** This classifier is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the features and target.

        Returns
        -------
        fitted_classifier:
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """

    @abstractmethod
    def predict(
        self,
        dataset: Table | TabularDataset,
    ) -> TabularDataset:
        """
        Predict the target values on the given dataset.

        Parameters
        ----------
        dataset:
            The dataset containing the feature vectors.

        Returns
        -------
        table:
            A dataset containing the given features and the predicted target.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_metrics(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> Table:
        """
        Summarize the classifier's metrics on the given data.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        metrics:
            A table containing the classifier's metrics.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.summarize(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def accuracy(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the accuracy of the classifier on the given data.

        The accuracy is the proportion of correctly predicted target values.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        accuracy:
            The classifier's accuracy.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.accuracy(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def f1_score(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> float:
        """
        Compute the classifier's $F_1$ score on the given data.

        The $F_1$ score is the harmonic mean of precision and recall.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        f1_score:
            The classifier's $F_1$ score.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.f1_score(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def precision(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> float:
        """
        Compute the classifier's precision on the given data.

        The precision is the proportion of positive predictions that were correct.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        precision:
            The classifier's precision.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.precision(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def recall(self, validation_or_test_set: Table | TabularDataset, positive_class: Any) -> float:
        """
        Compute the classifier's recall on the given data.

        The recall is the proportion of actual positives that were predicted correctly.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        recall:
            The classifier's recall.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.recall(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Classifier.
        """


def _extract_table(table_or_dataset: Table | TabularDataset) -> Table:
    """Extract the table from the given table or dataset."""
    if isinstance(table_or_dataset, TabularDataset):
        return table_or_dataset.to_table()
    else:
        return table_or_dataset
