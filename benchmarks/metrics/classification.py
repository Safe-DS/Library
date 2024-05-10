from __future__ import annotations

from timeit import timeit

import polars as pl

from typing import TYPE_CHECKING

from benchmarks.table.utils import create_synthetic_table
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.ml.classical.classification import Classifier
from safeds.ml.metrics import ClassificationMetrics

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class DummyClassifier(Classifier):
    """
    Dummy classifier to test metrics.

    Metrics methods expect a `TabularDataset` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TabularDataset) -> DummyClassifier:  # noqa: ARG002
        return self

    def predict(self, dataset: Table) -> TabularDataset:
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return dataset.to_tabular_dataset(target_name="predicted")

    @property
    def is_fitted(self) -> bool:
        return True

    def get_feature_names(self) -> list[str]:
        return ["predicted"]

    def get_target_name(self) -> str:
        return "expected"

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        pass


REPETITIONS = 10


def _run_accuracy_old() -> None:
    DummyClassifier().accuracy(table)


def _run_accuracy_new() -> None:
    ClassificationMetrics.accuracy(table.get_column("predicted"), table.get_column("expected"))


if __name__ == "__main__":
    # Create a synthetic Table
    table = (
        create_synthetic_table(10000, 2)
            .rename_column("column_0", "predicted")
            .rename_column("column_1", "expected")
    )

    # Run the benchmarks
    timings: dict[str, float] = {
        "accuracy_old": timeit(
            _run_accuracy_old,
            number=REPETITIONS,
        ),
        "accuracy_new": timeit(
            _run_accuracy_new,
            number=REPETITIONS,
        ),
    }

    # Print the timings
    with pl.Config(
        tbl_rows=-1,
    ):
        print(
            Table(
                {
                    "method": list(timings.keys()),
                    "timing": list(timings.values()),
                }
            )
        )
