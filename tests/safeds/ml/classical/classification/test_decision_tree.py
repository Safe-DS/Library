import pytest
from syrupy import SnapshotAssertion

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import NotFittedError, OutOfBoundsError
from safeds.ml.classical.classification import DecisionTreeClassifier
from safeds.ml.hyperparameters import Choice
from tests.helpers import os_mac, skip_if_os


@pytest.fixture
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset("col1")


class TestMaxDepth:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(max_depth=2).fit(training_set)
        assert fitted_model.max_depth == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(max_depth=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.max_depth == 2

    @pytest.mark.parametrize("max_depth", [-1, 0, Choice(-1)], ids=["minus_one", "zero", "invalid_choice"])
    def test_should_raise_if_less_than_or_equal_to_0(self, max_depth: int | None | Choice[int | None]) -> None:
        with pytest.raises(OutOfBoundsError):
            DecisionTreeClassifier(max_depth=max_depth)


class TestMinSampleCountInLeaves:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(min_sample_count_in_leaves=2).fit(training_set)
        assert fitted_model.min_sample_count_in_leaves == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(min_sample_count_in_leaves=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.min_samples_leaf == 2

    @pytest.mark.parametrize(
        "min_sample_count_in_leaves",
        [-1, 0, Choice(-1)],
        ids=["minus_one", "zero", "invalid_choice"],
    )
    def test_should_raise_if_less_than_or_equal_to_0(self, min_sample_count_in_leaves: int | Choice[int]) -> None:
        with pytest.raises(OutOfBoundsError):
            DecisionTreeClassifier(min_sample_count_in_leaves=min_sample_count_in_leaves)


class TestPlot:
    def test_should_raise_if_model_is_not_fitted(self) -> None:
        model = DecisionTreeClassifier()
        with pytest.raises(NotFittedError):
            model.plot()

    def test_should_check_that_plot_image_is_same_as_snapshot(
        self,
        training_set: TabularDataset,
        snapshot_png_image: SnapshotAssertion,
    ) -> None:
        skip_if_os([os_mac])

        fitted_model = DecisionTreeClassifier().fit(training_set)
        image = fitted_model.plot()
        assert image == snapshot_png_image
