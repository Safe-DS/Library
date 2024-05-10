from __future__ import annotations

from typing import TYPE_CHECKING, Any

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnLengthMismatchError

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Column


class ClassificationMetrics:

    @staticmethod
    def accuracy(predicted: Column | TabularDataset, expected: Column | TabularDataset) -> float:
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_column_length(predicted, expected)

        if expected.number_of_rows == 0:
            return 1.0  # Everything was predicted correctly

        return expected._series.eq(predicted._series).sum() / expected.number_of_rows

    @staticmethod
    def f1_score(predicted: Column | TabularDataset, expected: Column | TabularDataset, positive_class: Any) -> float:
        predicted = _extract_target(predicted)
        expected = _extract_target(expected)
        _check_column_length(predicted, expected)

        true_positives = (expected._series.eq(positive_class) & predicted._series.eq(positive_class)).sum()
        false_positives = (expected._series.ne(positive_class) & predicted._series.eq(positive_class)).sum()
        false_negatives = (expected._series.eq(positive_class) & predicted._series.ne(positive_class)).sum()

        if true_positives + false_positives + false_negatives == 0:
            return 1.0  # Only true negatives

        return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

    @staticmethod
    def precision(predicted: Column | TabularDataset, expected: Column | TabularDataset, positive_class: Any) -> float:
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_column_length(predicted, expected)

        true_positives = (expected._series.eq(positive_class) & predicted._series.eq(positive_class)).sum()
        predicted_positives = predicted._series.eq(positive_class).sum()

        if predicted_positives == 0:
            return 1.0  # All positive predictions were correct

        return true_positives / predicted_positives

    @staticmethod
    def recall(predicted: Column | TabularDataset, expected: Column | TabularDataset, positive_class: Any) -> float:
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_column_length(predicted, expected)

        true_positives = (expected._series.eq(positive_class) & predicted._series.eq(positive_class)).sum()
        actual_positives = expected._series.eq(positive_class).sum()

        if actual_positives == 0:
            return 1.0  # All actual positives were predicted correctly

        return true_positives / actual_positives

    @staticmethod
    def summarize(predicted: Column | TabularDataset, expected: Column | TabularDataset, positive_class: Any) -> Table:
        """
        Summarize classification metrics on the given data.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the classifier.
        expected:
            The expected target values.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        metrics:
            A table containing the classifier's metrics.
        """
        accuracy = ClassificationMetrics.accuracy(predicted, expected)
        precision = ClassificationMetrics.precision(predicted, expected, positive_class)
        recall = ClassificationMetrics.recall(predicted, expected, positive_class)
        f1_score = ClassificationMetrics.f1_score(predicted, expected, positive_class)

        return Table(
            {
                "metric": ["accuracy", "precision", "recall", "f1_score"],
                "value": [accuracy, precision, recall, f1_score],
            },
        )


def _extract_target(column_or_dataset: Column | TabularDataset) -> Column:
    if isinstance(column_or_dataset, TabularDataset):
        return column_or_dataset.target
    else:
        return column_or_dataset


def _check_column_length(expected: Column, predicted: Column) -> None:
    if expected.number_of_rows != predicted.number_of_rows:
        ColumnLengthMismatchError("")  # TODO: pass list of columns to exception, let it handle the formatting
