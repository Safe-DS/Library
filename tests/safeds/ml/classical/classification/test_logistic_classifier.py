import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import LogisticClassifier


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestC:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = LogisticClassifier(c=2).fit(training_set)
        assert fitted_model.c == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = LogisticClassifier(c=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.C == 2

    def test_clone(self, training_set: TabularDataset) -> None:
        fitted_model = LogisticClassifier(c=2).fit(training_set)
        cloned_classifier = fitted_model._clone()
        assert isinstance(cloned_classifier, LogisticClassifier)
        assert cloned_classifier.c == fitted_model.c

    @pytest.mark.parametrize("c", [-1.0, 0.0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, c: float) -> None:
        with pytest.raises(OutOfBoundsError):
            LogisticClassifier(c=c)
