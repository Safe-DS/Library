from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from safeds.data.labeled.containers import TabularDataset, TimeSeriesDataset
from safeds.data.tabular.containers import Table, Column
from safeds.exceptions import ColumnLengthMismatchError


class ClassificationMetrics(ABC):
    """A collection of classification metrics."""

    @abstractmethod
    def __init__(self) -> None: ...

    @staticmethod
    def summarize(predicted: Column | TabularDataset | TimeSeriesDataset, expected: Column | TabularDataset | TimeSeriesDataset, positive_class: Any) -> Table:
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
            A table containing the classification metrics.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

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

    @staticmethod
    def accuracy(predicted: Column | TabularDataset | TimeSeriesDataset, expected: Column | TabularDataset | TimeSeriesDataset) -> float:
        """
        Compute the accuracy on the given data.

        The accuracy is the proportion of predicted target values that were correct. The **higher** the accuracy, the
        better. Results range from 0.0 to 1.0.

        Parameters
        ----------
        predicted:
            The predicted target values produced by the classifier.
        expected:
            The expected target values.

        Returns
        -------
        accuracy:
            The calculated accuracy.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        from polars.exceptions import ComputeError

        if expected.row_count == 0:
            return 1.0  # Everything was predicted correctly (since there is nothing to predict)

        # For TimeSeries Predictions, where the output is a list of values.
        # Expected results are internally converted to a column containing multiple Columns for each prediction window
        # Currently only used in fit_by_exhaustive_search, where prediction metrics have to be calculated internally.
        if isinstance(expected.get_value(0), Column):
            sum_of_accuracies = 0.0
            for i in range(0, expected.row_count):
                predicted_row_as_col: Column = Column("predicted", predicted[i])
                expected_row_as_col = expected.get_value(i)
                sum_of_accuracies += ClassificationMetrics.accuracy(
                    predicted_row_as_col,
                    expected_row_as_col)
            return sum_of_accuracies / expected.row_count

        try:
            return expected._series.eq_missing(predicted._series).mean()
        except ComputeError:
            return 0.0  # Types are not compatible, so no prediction can be correct

    @staticmethod
    def f1_score(predicted: Column | TabularDataset | TimeSeriesDataset, expected: Column | TabularDataset | TimeSeriesDataset, positive_class: Any) -> float:
        """
        Compute the F₁ score on the given data.

        The F₁ score is the harmonic mean of precision and recall. The **higher** the F₁ score, the better the
        classifier. Results range from 0.0 to 1.0.

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
        f1_score:
            The calculated F₁ score.
        """
        predicted = _extract_target(predicted)
        expected = _extract_target(expected)
        _check_equal_length(predicted, expected)

        # For TimeSeries Predictions, where the output is a list of values.
        # Expected results are internally converted to a column containing multiple Columns for each prediction window
        # Currently only used in fit_by_exhaustive_search, where prediction metrics have to be calculated internally.
        if isinstance(expected.get_value(0), Column):
            sum_of_f1scores = 0.0
            for i in range(0, expected.row_count):
                predicted_row_as_col: Column = Column("predicted", predicted[i])
                expected_row_as_col = expected.get_value(i)
                sum_of_f1scores += ClassificationMetrics.f1_score(
                    predicted_row_as_col,
                    expected_row_as_col, positive_class)
            return sum_of_f1scores / expected.row_count

        true_positives = (expected._series.eq(positive_class) & predicted._series.eq(positive_class)).sum()
        false_positives = (expected._series.ne(positive_class) & predicted._series.eq(positive_class)).sum()
        false_negatives = (expected._series.eq(positive_class) & predicted._series.ne(positive_class)).sum()

        if true_positives + false_positives + false_negatives == 0:
            return 1.0  # Only true negatives (so all predictions are correct)

        return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

    @staticmethod
    def precision(predicted: Column | TabularDataset | TimeSeriesDataset, expected: Column | TabularDataset | TimeSeriesDataset, positive_class: Any) -> float:
        """
        Compute the precision on the given data.

        The precision is the proportion of positive predictions that were correct. The **higher** the precision, the
        better the classifier. Results range from 0.0 to 1.0.

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
        precision:
            The calculated precision.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        # For TimeSeries Predictions, where the output is a list of values.
        # Expected results are internally converted to a column containing multiple Columns for each prediction window
        # Currently only used in fit_by_exhaustive_search, where prediction metrics have to be calculated internally.
        if isinstance(expected.get_value(0), Column):
            sum_of_precisions = 0.0
            for i in range(0, expected.row_count):
                predicted_row_as_col: Column = Column("predicted", predicted[i])
                expected_row_as_col = expected.get_value(i)
                sum_of_precisions += ClassificationMetrics.precision(
                    predicted_row_as_col,
                    expected_row_as_col, positive_class)
            return sum_of_precisions / expected.row_count
        true_positives = (expected._series.eq(positive_class) & predicted._series.eq(positive_class)).sum()
        predicted_positives = predicted._series.eq(positive_class).sum()

        if predicted_positives == 0:
            return 1.0  # All positive predictions were correct (since there are none)

        return true_positives / predicted_positives

    @staticmethod
    def recall(predicted: Column | TabularDataset | TimeSeriesDataset, expected: Column | TabularDataset | TimeSeriesDataset, positive_class: Any) -> float:
        """
        Compute the recall on the given data.

        The recall is the proportion of actual positives that were predicted correctly. The **higher** the recall, the
        better the classifier. Results range from 0.0 to 1.0.

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
        recall:
            The calculated recall.
        """
        expected = _extract_target(expected)
        predicted = _extract_target(predicted)
        _check_equal_length(predicted, expected)

        # For TimeSeries Predictions, where the output is a list of values.
        # Expected results are internally converted to a column containing multiple Columns for each prediction window
        # Currently only used in fit_by_exhaustive_search, where prediction metrics have to be calculated internally.
        if isinstance(expected.get_value(0), Column):
            sum_of_recalls = 0.0
            for i in range(0, expected.row_count):
                predicted_row_as_col: Column = Column("predicted", predicted[i])
                expected_row_as_col = expected.get_value(i)
                sum_of_recalls += ClassificationMetrics.recall(
                    predicted_row_as_col,
                    expected_row_as_col, positive_class)
            return sum_of_recalls / expected.row_count

        true_positives = (expected._series.eq(positive_class) & predicted._series.eq(positive_class)).sum()
        actual_positives = expected._series.eq(positive_class).sum()

        if actual_positives == 0:
            return 1.0  # All actual positives were predicted correctly (since there are none)

        return true_positives / actual_positives


def _extract_target(column_or_dataset: Column | TabularDataset | TimeSeriesDataset) -> Column:
    """Extract the target column from the given column or dataset."""
    if isinstance(column_or_dataset, TabularDataset | TimeSeriesDataset):
        return column_or_dataset.target
    else:
        return column_or_dataset


# TODO: collect validation in one place?
def _check_equal_length(column1: Column, column2: Column) -> None:
    """
    Check if the columns have the same length and raise an error if they do not.

    Parameters
    ----------
    column1:
        The first column.
    column2:
        The second column.

    Raises
    ------
    ValueError
        If the columns have different lengths.
    """
    if column1.row_count != column2.row_count:
        ColumnLengthMismatchError("")  # TODO: pass list of columns to exception, let it handle the formatting
