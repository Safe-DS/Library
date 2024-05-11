from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import ModelNotFittedError
from safeds.ml.classical import SupervisedModel
from safeds.ml.metrics import ClassificationMetrics

if TYPE_CHECKING:
    from typing import Any

    from safeds.data.tabular.containers import Table


class Classifier(SupervisedModel, ABC):
    """A model for classification tasks."""

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

        **Note:** The model must be fitted.

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
            If the classifier has not been fitted yet.
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

        The accuracy is the proportion of predicted target values that were correct. The **higher** the accuracy, the
        better. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

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
            If the classifier has not been fitted yet.
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
        Compute the classifier's F₁ score on the given data.

        The F₁ score is the harmonic mean of precision and recall. The **higher** the F₁ score, the better the
        classifier. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        f1_score:
            The classifier's F₁ score.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
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

        The precision is the proportion of positive predictions that were correct. The **higher** the precision, the
        better the classifier. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

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
            If the classifier has not been fitted yet.
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

        The recall is the proportion of actual positives that were predicted correctly. The **higher** the recall, the
        better the classifier. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

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
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.recall(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )


def _extract_table(table_or_dataset: Table | TabularDataset) -> Table:
    """Extract the table from the given table or dataset."""
    if isinstance(table_or_dataset, TabularDataset):
        return table_or_dataset.to_table()
    else:
        return table_or_dataset
