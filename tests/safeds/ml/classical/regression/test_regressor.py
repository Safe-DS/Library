from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Self

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    ColumnLengthMismatchError,
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    FittingWithChoiceError,
    FittingWithoutChoiceError,
    LearningError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    PlainTableError,
)
from safeds.ml.classical.regression import (
    AdaBoostRegressor,
    DecisionTreeRegressor,
    ElasticNetRegressor,
    GradientBoostingRegressor,
    KNearestNeighborsRegressor,
    LassoRegressor,
    LinearRegressor,
    RandomForestRegressor,
    Regressor,
    RidgeRegressor,
    SupportVectorRegressor,
)
from safeds.ml.classical.regression._regressor import _check_metrics_preconditions
from safeds.ml.hyperparameters import Choice
from safeds.ml.metrics import RegressorMetric

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from sklearn.base import ClassifierMixin, RegressorMixin


def regressors() -> list[Regressor]:
    """
    Return the list of regressors to test.

    After you implemented a new regressor, add it to this list to ensure its `fit` and `predict` method work as
    expected. Place tests of methods that are specific to your regressor in a separate test file.

    Returns
    -------
    regressors : list[Regressor]
        The list of regressors to test.
    """
    return [
        AdaBoostRegressor(),
        DecisionTreeRegressor(),
        ElasticNetRegressor(),
        GradientBoostingRegressor(),
        KNearestNeighborsRegressor(2),
        LassoRegressor(),
        LinearRegressor(),
        RandomForestRegressor(),
        RidgeRegressor(),
        SupportVectorRegressor(),
    ]


def regressors_with_choices() -> list[Regressor]:
    """
    Return the list of regressors with Choices as Parameters to test choice functionality.

    After you implemented a new regressor, add it to this list to ensure its `fit_by_exhaustive_search` method works as
    expected. Place tests of methods that are specific to your regressor in a separate test file.

    Returns
    -------
    regressors : list[Regressor]
        The list of regressors to test.
    """
    return [
        AdaBoostRegressor(max_learner_count=Choice(1, 2), learning_rate=Choice(0.1, 0.2)),
        DecisionTreeRegressor(max_depth=Choice(1, 2), min_sample_count_in_leaves=Choice(1, 2)),
        GradientBoostingRegressor(tree_count=Choice(1, 2), learning_rate=Choice(0.1, 0.2)),
        KNearestNeighborsRegressor(neighbor_count=Choice(1, 2)),
        RandomForestRegressor(tree_count=Choice(1, 2), max_depth=Choice(1, 2), min_sample_count_in_leaves=Choice(1, 2)),
        SupportVectorRegressor(c=Choice(0.5, 1.0)),
    ]


@pytest.fixture()
def valid_data() -> TabularDataset:
    return Table(
        {
            "id": [1, 4, 7, 10],
            "feat1": [2, 5, 8, 11],
            "feat2": [3, 6, 9, 12],
            "target": [0, 1, 0, 1],
        },
    ).to_tabular_dataset(target_name="target", extra_names=["id"])


@pytest.mark.parametrize("regressor_with_choice", regressors_with_choices(), ids=lambda x: x.__class__.__name__)
class TestChoiceRegressors:

    def test_workflow_with_choice_parameter(self, regressor_with_choice: Regressor, valid_data: TabularDataset) -> None:
        model = regressor_with_choice.fit_by_exhaustive_search(valid_data, RegressorMetric.MEAN_SQUARED_ERROR)
        assert isinstance(model, type(regressor_with_choice))
        pred = model.predict(valid_data)
        assert isinstance(pred, TabularDataset)

    def test_should_raise_if_model_is_fitted_with_choice(
        self, regressor_with_choice: Regressor, valid_data: TabularDataset,
    ) -> None:
        with pytest.raises(FittingWithChoiceError):
            regressor_with_choice.fit(valid_data)


class TestFitByExhaustiveSearch:
    @pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_model_is_fitted_by_exhaustive_search_without_choice(
        self, regressor: Regressor, valid_data: TabularDataset,
    ) -> None:
        with pytest.raises(FittingWithoutChoiceError):
            regressor.fit_by_exhaustive_search(valid_data, optimization_metric=RegressorMetric.MEAN_SQUARED_ERROR)

    def test_should_raise_if_model_is_fitted_by_exhaustive_search_with_empty_choice(
        self, valid_data: TabularDataset,
    ) -> None:
        with pytest.raises(LearningError):
            AdaBoostRegressor(max_learner_count=Choice(), learning_rate=Choice()).fit_by_exhaustive_search(
                valid_data, optimization_metric=RegressorMetric.MEAN_SQUARED_ERROR,
            )


@pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
class TestFit:
    def test_should_succeed_on_valid_data(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        regressor.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_not_change_input_regressor(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        regressor.fit(valid_data)
        assert not regressor.is_fitted

    def test_should_not_change_input_table(self, regressor: Regressor, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        regressor.fit(valid_data)
        assert valid_data == valid_data_copy

    @pytest.mark.parametrize(
        ("invalid_data", "expected_error", "expected_error_msg"),
        [
            (
                Table(
                    {
                        "id": [1, 4],
                        "feat1": ["a", 5],
                        "feat2": [3, 6],
                        "target": [0, 1],
                    },
                ).to_tabular_dataset(target_name="target", extra_names=["id"]),
                NonNumericColumnError,
                r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\{'feat1'\}",
            ),
            (
                Table(
                    {
                        "id": [1, 4],
                        "feat1": [None, 5],
                        "feat2": [3, 6],
                        "target": [0, 1],
                    },
                ).to_tabular_dataset(target_name="target", extra_names=["id"]),
                MissingValuesColumnError,
                r"Tried to do an operation on one or multiple columns containing missing values: \n\{'feat1'\}",
            ),
            (
                Table(
                    {
                        "id": [],
                        "feat1": [],
                        "feat2": [],
                        "target": [],
                    },
                ).to_tabular_dataset(target_name="target", extra_names=["id"]),
                DatasetMissesDataError,
                r"Dataset contains no rows",
            ),
        ],
        ids=["non-numerical data", "missing values in data", "no rows in data"],
    )
    def test_should_raise_on_invalid_data(
        self,
        regressor: Regressor,
        invalid_data: TabularDataset,
        expected_error: Any,
        expected_error_msg: str,
    ) -> None:
        with pytest.raises(expected_error, match=expected_error_msg):
            regressor.fit(invalid_data)

    @pytest.mark.parametrize(
        "table",
        [
            Table(
                {
                    "a": [1.0, 0.0, 0.0, 0.0],
                    "b": [0.0, 1.0, 1.0, 0.0],
                    "c": [0.0, 0.0, 0.0, 1.0],
                },
            ),
        ],
        ids=["table"],
    )
    def test_should_raise_if_given_normal_table(self, regressor: Regressor, table: Table) -> None:
        with pytest.raises(PlainTableError):
            regressor.fit(table)  # type: ignore[arg-type]


@pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
class TestPredict:
    def test_should_include_features_of_input_table(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_input_table(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_set_correct_target_name(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_not_change_input_table(self, regressor: Regressor, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        fitted_classifier = regressor.fit(valid_data)
        fitted_classifier.predict(valid_data.features)
        assert valid_data == valid_data_copy

    def test_should_raise_if_not_fitted(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        with pytest.raises(ModelNotFittedError):
            regressor.predict(valid_data.features)

    def test_should_raise_if_dataset_misses_features(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        fitted_regressor = regressor.fit(valid_data)
        with pytest.raises(DatasetMissesFeaturesError, match="[feat1, feat2]"):
            fitted_regressor.predict(valid_data.features.remove_columns(["feat1", "feat2"]))

    @pytest.mark.parametrize(
        ("invalid_data", "expected_error", "expected_error_msg"),
        [
            (
                Table(
                    {
                        "id": [1, 4],
                        "feat1": ["a", 5],
                        "feat2": [3, 6],
                    },
                ),
                NonNumericColumnError,
                (
                    r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\{'feat1'\}\nYou"
                    r" can use the LabelEncoder or OneHotEncoder to transform your non-numerical data to numerical"
                    r" data.\nThe OneHotEncoder should be used if you work with nominal data. If your data contains too"
                    r" many different values\nor is ordinal, you should use the LabelEncoder."
                ),
            ),
            (
                Table(
                    {
                        "id": [1, 4],
                        "feat1": [None, 5],
                        "feat2": [3, 6],
                    },
                ),
                MissingValuesColumnError,
                (
                    r"Tried to do an operation on one or multiple columns containing missing values: \n\{'feat1'\}\nYou"
                    r" can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
                    r" remove the missing values entirely you can use the method"
                    r" `Table.remove_rows_with_missing_values`."
                ),
            ),
            (
                Table(
                    {
                        "id": [],
                        "feat1": [],
                        "feat2": [],
                    },
                ),
                DatasetMissesDataError,
                r"Dataset contains no rows",
            ),
        ],
        ids=["non-numerical data", "missing values in data", "no rows in data"],
    )
    def test_should_raise_on_invalid_data(
        self,
        regressor: Regressor,
        valid_data: TabularDataset,
        invalid_data: Table,
        expected_error: Any,
        expected_error_msg: str,
    ) -> None:
        regressor = regressor.fit(valid_data)
        with pytest.raises(expected_error, match=expected_error_msg):
            regressor.predict(invalid_data)


@pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
class TestIsFitted:
    def test_should_return_false_before_fitting(self, regressor: Regressor) -> None:
        assert not regressor.is_fitted

    def test_should_return_true_after_fitting(self, regressor: Regressor, valid_data: TabularDataset) -> None:
        fitted_regressor = regressor.fit(valid_data)
        assert fitted_regressor.is_fitted


class TestHash:
    @pytest.mark.parametrize(
        ("regressor1", "regressor2"),
        ([(x, y) for x in regressors() for y in regressors() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_regressor(self, regressor1: Regressor, regressor2: Regressor) -> None:
        assert hash(regressor1) == hash(regressor2)

    @pytest.mark.parametrize(
        ("regressor1", "regressor2"),
        ([(x, y) for x in regressors() for y in regressors() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_regressor(
        self,
        regressor1: Regressor,
        regressor2: Regressor,
    ) -> None:
        assert hash(regressor1) != hash(regressor2)

    @pytest.mark.parametrize("regressor1", regressors(), ids=lambda x: x.__class__.__name__)
    def test_should_return_different_hash_for_same_regressor_fit(
        self,
        regressor1: Regressor,
        valid_data: TabularDataset,
    ) -> None:
        regressor1_fit = regressor1.fit(valid_data)
        assert hash(regressor1) != hash(regressor1_fit)

    @pytest.mark.parametrize(
        ("regressor1", "regressor2"),
        (list(itertools.product(regressors(), regressors()))),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_regressor_fit(
        self,
        regressor1: Regressor,
        regressor2: Regressor,
        valid_data: TabularDataset,
    ) -> None:
        regressor1_fit = regressor1.fit(valid_data)
        assert hash(regressor1_fit) != hash(regressor2)


class DummyRegressor(Regressor):
    """
    Dummy regressor to test metrics.

    Metrics methods expect a `TabularDataset` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def __init__(self) -> None:
        super().__init__()

        self._target_name = "expected"

    def __hash__(self) -> int:
        raise NotImplementedError

    def _clone(self) -> Self:
        return self

    def _get_sklearn_model(self) -> ClassifierMixin | RegressorMixin:
        pass

    def fit(self, _training_set: TabularDataset) -> DummyRegressor:
        return self

    def predict(self, dataset: Table | TabularDataset) -> TabularDataset:
        if isinstance(dataset, TabularDataset):
            dataset = dataset.to_table()

        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return dataset.to_tabular_dataset(target_name="predicted")

    @property
    def is_fitted(self) -> bool:
        return True


class TestSummarizeMetrics:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            (
                [1, 2],
                [1, 2],
                Table(
                    {
                        "metric": [
                            "coefficient_of_determination",
                            "mean_absolute_error",
                            "mean_squared_error",
                            "median_absolute_deviation",
                        ],
                        "value": [
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                    },
                ),
            ),
        ],
    )
    def test_valid_data(self, predicted: list[float], expected: list[float], result: Table) -> None:
        table = Table(
            {
                "predicted": predicted,
                "expected": expected,
            },
        ).to_tabular_dataset(
            target_name="expected",
        )

        assert DummyRegressor().summarize_metrics(table) == result


class TestMeanAbsoluteError:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            ([1, 2], [1, 2], 0),
            ([0, 0], [1, 1], 1),
            ([1, 1, 1], [2, 2, 11], 4),
            ([0, 0, 0], [10, 2, 18], 10),
            ([0.5, 0.5], [1.5, 1.5], 1),
        ],
    )
    def test_valid_data(self, predicted: list[float], expected: list[float], result: float) -> None:
        table = Table(
            {
                "predicted": predicted,
                "expected": expected,
            },
        ).to_tabular_dataset(
            target_name="expected",
        )

        assert DummyRegressor().mean_absolute_error(table) == result


class TestMeanSquaredError:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [([1, 2], [1, 2], 0), ([0, 0], [1, 1], 1), ([1, 1, 1], [2, 2, 11], 34)],
        ids=["perfect_prediction", "bad_prediction", "worst_prediction"],
    )
    def test_valid_data(self, predicted: list[float], expected: list[float], result: float) -> None:
        table = Table({"predicted": predicted, "expected": expected}).to_tabular_dataset(
            target_name="expected",
        )

        assert DummyRegressor().mean_squared_error(table) == result


class TestCheckMetricsPreconditions:
    @pytest.mark.parametrize(
        ("actual", "expected", "error"),
        [
            (["A", "B"], [1, 2], TypeError),
            ([1, 2], ["A", "B"], TypeError),
            ([1, 2, 3], [1, 2], ColumnLengthMismatchError),
        ],
    )
    def test_should_raise_if_validation_fails(
        self,
        actual: list[str | int],
        expected: list[str | int],
        error: type[Exception],
    ) -> None:
        actual_column: Column = Column("actual", actual)
        expected_column: Column = Column("expected", expected)
        with pytest.raises(error):
            _check_metrics_preconditions(actual_column, expected_column)
