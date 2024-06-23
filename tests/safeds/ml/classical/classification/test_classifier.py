from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Self

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    PlainTableError, FittingWithoutChoiceError, LearningError, FittingWithChoiceError,
)
from safeds.ml.classical.classification import (
    AdaBoostClassifier,
    Classifier,
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    KNearestNeighborsClassifier,
    LogisticClassifier,
    RandomForestClassifier,
    SupportVectorClassifier,
)
from safeds.ml.hyperparameters import Choice
from safeds.ml.metrics import ClassifierMetric

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from sklearn.base import ClassifierMixin


def classifiers() -> list[Classifier]:
    """
    Return the list of classifiers to test.

    After you implemented a new classifier, add it to this list to ensure its `fit` and `predict` method work as
    expected. Place tests of methods that are specific to your classifier in a separate test file.

    Returns
    -------
    classifiers : list[Classifier]
        The list of classifiers to test.
    """
    return [
        AdaBoostClassifier(),
        DecisionTreeClassifier(),
        GradientBoostingClassifier(),
        KNearestNeighborsClassifier(2),
        LogisticClassifier(),
        RandomForestClassifier(),
        SupportVectorClassifier(),
    ]


def classifiers_with_choices() -> list[Classifier]:
    """
    Return the list of classifiers with Choices as Parameters to test choice functionality.

    After you implemented a new classifier, add it to this list to ensure its `fit_by_exhaustive_search` method works as
    expected. Place tests of methods that are specific to your classifier in a separate test file.

    Returns
    -------
    classifiers : list[Classifier]
        The list of classifiers to test.
    """
    return [
        AdaBoostClassifier(max_learner_count=Choice(1, 2), learning_rate=Choice(0.1, 0.2)),
        DecisionTreeClassifier(max_depth=Choice(1, 2), min_sample_count_in_leaves=Choice(1, 2)),
        GradientBoostingClassifier(tree_count=Choice(1, 2), learning_rate=Choice(0.1, 0.2)),
        KNearestNeighborsClassifier(neighbor_count=Choice(1, 2)),
        RandomForestClassifier(tree_count=Choice(1, 2), max_depth=Choice(1, 2),
                               min_sample_count_in_leaves=Choice(1, 2)),
        SupportVectorClassifier(c=Choice(0.5, 1.0)),
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


@pytest.mark.parametrize("classifier_with_choice", classifiers_with_choices(), ids=lambda x: x.__class__.__name__)
class TestChoiceClassifiers:

    def test_should_raise_if_no_positive_class_is_provided(self, classifier_with_choice: Classifier,
                                                           valid_data: TabularDataset) -> None:
        with pytest.raises(LearningError):
            classifier_with_choice.fit_by_exhaustive_search(valid_data, optimization_metric=ClassifierMetric.PRECISION)

    def test_workflow_with_choice_parameter(self, classifier_with_choice: Classifier, valid_data: TabularDataset) -> None:
        model = classifier_with_choice.fit_by_exhaustive_search(valid_data, ClassifierMetric.ACCURACY)
        assert isinstance(model, type(classifier_with_choice))
        pred = model.predict(valid_data)
        assert isinstance(pred, TabularDataset)

    def test_should_raise_if_model_is_fitted_with_choice(self, classifier_with_choice: Classifier,
                                                         valid_data: TabularDataset) -> None:
        with pytest.raises(FittingWithChoiceError):
            classifier_with_choice.fit(valid_data)


class TestFitByExhaustiveSearch:

    @pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_model_is_fitted_by_exhaustive_search_without_choice(self,
                                                                                 classifier: Classifier,
                                                                                 valid_data: TabularDataset) -> None:
        with pytest.raises(FittingWithoutChoiceError):
            classifier.fit_by_exhaustive_search(valid_data, optimization_metric=ClassifierMetric.ACCURACY)

    def test_should_raise_if_model_is_fitted_by_exhaustive_search_with_empty_choice(self,
                                                                                    valid_data: TabularDataset) -> None:
        with pytest.raises(LearningError):
            AdaBoostClassifier(max_learner_count=Choice(), learning_rate=Choice()).fit_by_exhaustive_search(valid_data,
                                                                                                            optimization_metric=ClassifierMetric.ACCURACY)


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestFit:
    def test_should_succeed_on_valid_data(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        classifier.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_not_change_input_classifier(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        classifier.fit(valid_data)
        assert not classifier.is_fitted

    def test_should_not_change_input_table(self, classifier: Classifier, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        classifier.fit(valid_data)
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
                        "target": [0, 1],
                    },
                ).to_tabular_dataset(target_name="target", extra_names=["id"]),
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
        classifier: Classifier,
        invalid_data: TabularDataset,
        expected_error: Any,
        expected_error_msg: str,
    ) -> None:
        with pytest.raises(expected_error, match=expected_error_msg):
            classifier.fit(invalid_data)

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
    def test_should_raise_if_given_normal_table(self, classifier: Classifier, table: Table) -> None:
        with pytest.raises(PlainTableError):
            classifier.fit(table)  # type: ignore[arg-type]


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestPredict:
    def test_should_include_features_of_input_table(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_input_table(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        fitted_regressor = classifier.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_set_correct_target_name(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_not_change_input_table(self, classifier: Classifier, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        fitted_classifier = classifier.fit(valid_data)
        fitted_classifier.predict(valid_data.features)
        assert valid_data == valid_data_copy

    def test_should_raise_if_not_fitted(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        with pytest.raises(ModelNotFittedError):
            classifier.predict(valid_data.features)

    def test_should_raise_if_dataset_misses_features(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(DatasetMissesFeaturesError, match="[feat1, feat2]"):
            fitted_classifier.predict(valid_data.features.remove_columns(["feat1", "feat2"]))

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
                r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\{'feat1'\}",
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
                r"Tried to do an operation on one or multiple columns containing missing values: \n\{'feat1'\}",
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
        classifier: Classifier,
        valid_data: TabularDataset,
        invalid_data: Table,
        expected_error: Any,
        expected_error_msg: str,
    ) -> None:
        classifier = classifier.fit(valid_data)
        with pytest.raises(expected_error, match=expected_error_msg):
            classifier.predict(invalid_data)


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestIsFitted:
    def test_should_return_false_before_fitting(self, classifier: Classifier) -> None:
        assert not classifier.is_fitted

    def test_should_return_true_after_fitting(self, classifier: Classifier, valid_data: TabularDataset) -> None:
        fitted_classifier = classifier.fit(valid_data)
        assert fitted_classifier.is_fitted


class TestHash:
    @pytest.mark.parametrize(
        ("classifier1", "classifier2"),
        ([(x, y) for x in classifiers() for y in classifiers() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_classifier(
        self,
        classifier1: Classifier,
        classifier2: Classifier,
    ) -> None:
        assert hash(classifier1) == hash(classifier2)

    @pytest.mark.parametrize(
        ("classifier1", "classifier2"),
        ([(x, y) for x in classifiers() for y in classifiers() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_classifier(
        self,
        classifier1: Classifier,
        classifier2: Classifier,
    ) -> None:
        assert hash(classifier1) != hash(classifier2)

    @pytest.mark.parametrize("classifier1", classifiers(), ids=lambda x: x.__class__.__name__)
    def test_should_return_different_hash_for_same_classifier_fit(
        self,
        classifier1: Classifier,
        valid_data: TabularDataset,
    ) -> None:
        regressor1_fit = classifier1.fit(valid_data)
        assert hash(classifier1) != hash(regressor1_fit)

    @pytest.mark.parametrize(
        ("classifier1", "classifier2"),
        (list(itertools.product(classifiers(), classifiers()))),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_classifier_fit(
        self,
        classifier1: Classifier,
        classifier2: Classifier,
        valid_data: TabularDataset,
    ) -> None:
        classifier1_fit = classifier1.fit(valid_data)
        assert hash(classifier1_fit) != hash(classifier2)


class DummyClassifier(Classifier):
    """
    Dummy classifier to test metrics.

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

    def fit(self, training_set: TabularDataset) -> DummyClassifier:  # noqa: ARG002
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

    def _get_sklearn_model(self) -> ClassifierMixin:
        pass


class TestSummarizeMetrics:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            (
                [1, 2],
                [1, 2],
                Table(
                    {
                        "metric": ["accuracy", "precision", "recall", "f1_score"],
                        "value": [1.0, 1.0, 1.0, 1.0],
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

        assert DummyClassifier().summarize_metrics(table, 1) == result


class TestAccuracy:
    def test_with_same_type(self) -> None:
        table = Table(
            {
                "predicted": [1, 2, 3, 4],
                "expected": [1, 2, 3, 3],
            },
        ).to_tabular_dataset(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.75

    def test_with_different_types(self) -> None:
        table = Table(
            {
                "predicted": ["1", "2", "3", "4"],
                "expected": [1, 2, 3, 3],
            },
        ).to_tabular_dataset(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.0


class TestPrecision:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            (
                [2, 0, 0, 0],
                [0, 1, 1, 2],
                1.0,
            ),
            (
                [2, 1, 1, 0],
                [0, 1, 1, 2],
                1.0,
            ),
            (
                [2, 1, 1, 0],
                [0, 1, 0, 2],
                0.5,
            ),
            (
                [2, 1, 1, 0],
                [0, 0, 0, 1],
                0.0,
            ),
        ],
        ids=[
            "no positive predictions",
            "all correct positive predictions",
            "some correct positive predictions",
            "no correct positive predictions",
        ],
    )
    def test_should_compute_precision(self, predicted: list, expected: list, result: float) -> None:
        table = Table(
            {
                "predicted": predicted,
                "expected": expected,
            },
        ).to_tabular_dataset(target_name="expected")

        assert DummyClassifier().precision(table, 1) == result


class TestRecall:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            (
                [2, 0, 0, 0],
                [0, 0, 0, 2],
                1.0,
            ),
            (
                [2, 1, 1, 0],
                [0, 1, 1, 2],
                1.0,
            ),
            (
                [2, 1, 0, 0],
                [0, 1, 1, 2],
                0.5,
            ),
            (
                [2, 1, 1, 0],
                [0, 0, 0, 1],
                0.0,
            ),
        ],
        ids=[
            "no positive expectations",
            "all positive expectations recalled",
            "some positive expectations recalled",
            "no positive expectations recalled",
        ],
    )
    def test_should_compute_recall(self, predicted: list, expected: list, result: float) -> None:
        table = Table(
            {
                "predicted": predicted,
                "expected": expected,
            },
        ).to_tabular_dataset(target_name="expected")

        assert DummyClassifier().recall(table, 1) == result


class TestF1Score:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            # From precision
            (
                [2, 0, 0, 0],
                [0, 1, 1, 2],
                0.0,
            ),
            (
                [2, 1, 1, 0],
                [0, 1, 1, 2],
                1.0,
            ),
            (
                [2, 1, 1, 0],
                [0, 1, 0, 2],
                2 / 3,
            ),
            (
                [2, 1, 1, 0],
                [0, 0, 0, 1],
                0.0,
            ),
            # From recall
            (
                [2, 0, 0, 0],
                [0, 0, 0, 2],
                1.0,
            ),
            (
                [2, 1, 0, 0],
                [0, 1, 1, 2],
                2 / 3,
            ),
        ],
        ids=[
            # From precision
            "no positive predictions",
            "all correct positive predictions",
            "some correct positive predictions",
            "no correct positive predictions",
            # From recall
            "no positive expectations",
            "some positive expectations recalled",
        ],
    )
    def test_should_compute_f1_score(self, predicted: list, expected: list, result: float) -> None:
        table = Table(
            {
                "predicted": predicted,
                "expected": expected,
            },
        ).to_tabular_dataset(target_name="expected")

        assert DummyClassifier().f1_score(table, 1) == result
