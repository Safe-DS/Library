from __future__ import annotations

from typing import TYPE_CHECKING, Any

from safeds.data.labeled.containers import TabularDataset

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import Column, Table


class ClassificationMetrics:

    @staticmethod
    def accuracy(expected: Column | TabularDataset, predicted: Column | TabularDataset):

        # if expected.number_of_rows != predicted.number_of_rows:


        if isinstance(expected, TabularDataset):
            expected = expected.target
        if isinstance(predicted, TabularDataset):
            predicted = predicted.target

        return expected._series.eq(predicted._series).sum() / len(expected)

    @staticmethod
    def confusion_matrix(expected: Column | TabularDataset, predicted: Column | TabularDataset) -> Image:
        if isinstance(expected, TabularDataset):
            expected = expected.target
        if isinstance(predicted, TabularDataset):
            predicted = predicted.target

        return confusion_matrix(expected, predicted)

    @staticmethod
    def f1_score(expected: Column | TabularDataset, predicted: Column | TabularDataset, positive_class: Any) -> float:
        if isinstance(expected, TabularDataset):
            expected = expected.target
        if isinstance(predicted, TabularDataset):
            predicted = predicted.target

        return f1_score(expected, predicted)

    @staticmethod
    def precision(expected: Column | TabularDataset, predicted: Column | TabularDataset, positive_class: Any) -> float:
        if isinstance(expected, TabularDataset):
            expected = expected.target
        if isinstance(predicted, TabularDataset):
            predicted = predicted.target

        return precision_score(expected, predicted)

    @staticmethod
    def recall(expected: Column | TabularDataset, predicted: Column | TabularDataset, positive_class: Any) -> float:
        if isinstance(expected, TabularDataset):
            expected = expected.target
        if isinstance(predicted, TabularDataset):
            predicted = predicted.target

        return recall_score(expected, predicted)

    @staticmethod
    def summarize(expected: Column | TabularDataset, predicted: Column | TabularDataset, positive_class: Any) -> Table:
        acc = ClassificationMetrics.accuracy(expected, predicted)
        prec = ClassificationMetrics.precision(expected, predicted)
        rec = ClassificationMetrics.recall(expected, predicted)
        f1 = ClassificationMetrics.f1(expected, predicted)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
