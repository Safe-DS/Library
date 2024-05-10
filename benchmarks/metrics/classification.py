from __future__ import annotations

from timeit import timeit
from typing import TYPE_CHECKING

import polars as pl

from benchmarks.table.utils import create_synthetic_table
from safeds.data.tabular.containers import Table
from safeds.ml.metrics import ClassificationMetrics


REPETITIONS = 10


def _run_accuracy() -> None:
    ClassificationMetrics.accuracy(table.get_column("predicted"), table.get_column("expected"))


def _run_f1_score() -> None:
    ClassificationMetrics.f1_score(table.get_column("predicted"), table.get_column("expected"), 1)


def _run_precision() -> None:
    ClassificationMetrics.precision(table.get_column("predicted"), table.get_column("expected"), 1)


def _run_recall() -> None:
    ClassificationMetrics.recall(table.get_column("predicted"), table.get_column("expected"), 1)


if __name__ == "__main__":
    # Create a synthetic Table
    table = (
        create_synthetic_table(10000, 2)
        .rename_column("column_0", "predicted")
        .rename_column("column_1", "expected")
    )

    # Run the benchmarks
    timings: dict[str, float] = {
        "accuracy": timeit(
            _run_accuracy,
            number=REPETITIONS,
        ),
        "f1_score": timeit(
            _run_f1_score,
            number=REPETITIONS,
        ),
        "precision": timeit(
            _run_precision,
            number=REPETITIONS,
        ),
        "recall": timeit(
            _run_recall,
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
