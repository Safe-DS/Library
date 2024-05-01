from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import pytest
from safeds.data.labeled.containers import TaggedTable
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    DatasetContainsTargetError,
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    UntaggedTableError,
)
from safeds.ml.classical.classification import (
    AdaBoostClassifier,
    Classifier,
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    KNearestNeighborsClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    SupportVectorMachineClassifier,
)

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
        LogisticRegressionClassifier(),
        RandomForestClassifier(),
        SupportVectorMachineClassifier(),
    ]


@pytest.fixture()
def valid_data() -> TaggedTable:
    return Table(
        {
            "id": [1, 4],
            "feat1": [2, 5],
            "feat2": [3, 6],
            "target": [0, 1],
        },
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestFit:
    def test_should_succeed_on_valid_data(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        classifier.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_not_change_input_classifier(self, classifier: Classifier, valid_data: TaggedTable) -> None:
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
                ).tag_columns(target_name="target", feature_names=["feat1", "feat2"]),
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
                ).tag_columns(target_name="target", feature_names=["feat1", "feat2"]),
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
                ).tag_columns(target_name="target", feature_names=["feat1", "feat2"]),
                DatasetMissesDataError,
                r"Dataset contains no rows",
            ),
        ],
        ids=["non-numerical data", "missing values in data", "no rows in data"],
    )
    def test_should_raise_on_invalid_data(
        self,
        classifier: Classifier,
        invalid_data: TaggedTable,
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
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, classifier: Classifier, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            classifier.fit(table)  # type: ignore[arg-type]


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestPredict:
    def test_should_include_features_of_input_table(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_input_table(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_regressor = classifier.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_set_correct_target_name(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_not_change_input_table(self, classifier: Classifier, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        fitted_classifier = classifier.fit(valid_data)
        fitted_classifier.predict(valid_data.features)
        assert valid_data == valid_data_copy

    def test_should_raise_if_not_fitted(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        with pytest.raises(ModelNotFittedError):
            classifier.predict(valid_data.features)

    def test_should_raise_if_dataset_contains_target(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(DatasetContainsTargetError, match="target"):
            fitted_classifier.predict(valid_data._as_table())

    def test_should_raise_if_dataset_misses_features(self, classifier: Classifier, valid_data: TaggedTable) -> None:
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
        valid_data: TaggedTable,
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

    def test_should_return_true_after_fitting(self, classifier: Classifier, valid_data: TaggedTable) -> None:
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
        valid_data: TaggedTable,
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
        valid_data: TaggedTable,
    ) -> None:
        classifier1_fit = classifier1.fit(valid_data)
        assert hash(classifier1_fit) != hash(classifier2)


class DummyClassifier(Classifier):
    """
    Dummy classifier to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> DummyClassifier:  # noqa: ARG002
        return self

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Library/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return dataset.tag_columns(target_name="predicted")

    @property
    def is_fitted(self) -> bool:
        return True

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        pass


class TestAccuracy:
    def test_with_same_type(self) -> None:
        table = Table(
            {
                "predicted": [1, 2, 3, 4],
                "expected": [1, 2, 3, 3],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.75

    def test_with_different_types(self) -> None:
        table = Table(
            {
                "predicted": ["1", "2", "3", "4"],
                "expected": [1, 2, 3, 3],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.0

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
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            DummyClassifier().accuracy(table)  # type: ignore[arg-type]


class TestPrecision:
    def test_should_compare_result(self) -> None:
        table = Table(
            {
                "predicted": [1, 1, 0, 2],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().precision(table, 1) == 0.5

    def test_should_compare_result_with_different_types(self) -> None:
        table = Table(
            {
                "predicted": [1, "1", "0", "2"],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().precision(table, 1) == 1.0

    def test_should_return_1_if_never_expected_to_be_positive(self) -> None:
        table = Table(
            {
                "predicted": ["lol", "1", "0", "2"],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().precision(table, 1) == 1.0

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
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            DummyClassifier().precision(table, 1)  # type: ignore[arg-type]


class TestRecall:
    def test_should_compare_result(self) -> None:
        table = Table(
            {
                "predicted": [1, 1, 0, 2],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().recall(table, 1) == 0.5

    def test_should_compare_result_with_different_types(self) -> None:
        table = Table(
            {
                "predicted": [1, "1", "0", "2"],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().recall(table, 1) == 0.5

    def test_should_return_1_if_never_expected_to_be_positive(self) -> None:
        table = Table(
            {
                "predicted": ["lol", "1", "0", "2"],
                "expected": [2, 0, 5, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().recall(table, 1) == 1.0

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
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            DummyClassifier().recall(table, 1)  # type: ignore[arg-type]


class TestF1Score:
    def test_should_compare_result(self) -> None:
        table = Table(
            {
                "predicted": [1, 1, 0, 2],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().f1_score(table, 1) == 0.5

    def test_should_compare_result_with_different_types(self) -> None:
        table = Table(
            {
                "predicted": [1, "1", "0", "2"],
                "expected": [1, 0, 1, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().f1_score(table, 1) == pytest.approx(0.6666667)

    def test_should_return_1_if_never_expected_or_predicted_to_be_positive(self) -> None:
        table = Table(
            {
                "predicted": ["lol", "1", "0", "2"],
                "expected": [2, 0, 2, 2],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().f1_score(table, 1) == 1.0

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
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            DummyClassifier().f1_score(table, 1)  # type: ignore[arg-type]
