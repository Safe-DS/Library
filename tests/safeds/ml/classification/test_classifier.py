from __future__ import annotations

import pandas as pd

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.ml.classification import Classifier


class DummyClassifier(Classifier):
    """
    Dummy classifier to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> DummyClassifier:
        # pylint: disable=unused-argument
        return self

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Stdlib/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return TaggedTable(dataset, target_name="predicted")


class TestAccuracy:
    def test_with_same_type(self) -> None:
        c1 = Column("predicted", pd.Series(data=[1, 2, 3, 4]))
        c2 = Column("expected", pd.Series(data=[1, 2, 3, 3]))
        table = TaggedTable(Table.from_columns([c1, c2]), target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.75

    def test_with_different_types(self) -> None:
        c1 = Column("predicted", pd.Series(data=["1", "2", "3", "4"]))
        c2 = Column("expected", pd.Series(data=[1, 2, 3, 3]))
        table = TaggedTable(Table.from_columns([c1, c2]), target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.0
