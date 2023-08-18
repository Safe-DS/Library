import warnings
from typing import Any

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.classical._util_sklearn import fit, predict
from safeds.ml.classical.regression import LinearRegression


def test_predict_should_not_warn_about_feature_names() -> None:
    """See https://github.com/Safe-DS/Library/issues/51."""
    training_set = Table({"a": [1, 2, 3], "b": [2, 4, 6]}).tag_columns(target_name="b")

    model = LinearRegression()
    fitted_model = model.fit(training_set)

    test_set = Table({"a": [4, 5, 6]})

    # No warning should be emitted
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X has feature names")
        fitted_model.predict(test_set)


class MLModelRaiseValueErrorOnFitAndPredict:
    x, y = None, None

    def fit(self, x: Any, y: Any) -> None:
        # The Linter does not want unnecessary parameters, so we just assign them to the class values
        self.x = x
        self.y = y
        raise ValueError("Raise ValueError (LearningError) in fit for Test")

    def predict(self, x: Any) -> None:
        # The Linter does not want unnecessary parameters, so we just assign it to the class value
        self.x = x
        raise ValueError("Raise ValueError (PredictionError) in predict for Test")


def test_should_raise_learning_error() -> None:
    tagged_table = Table({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]}).tag_columns("col3")
    with pytest.raises(
        LearningError,
        match=r"Error occurred while learning: Raise ValueError \(LearningError\) in fit for Test",
    ):
        fit(MLModelRaiseValueErrorOnFitAndPredict(), tagged_table)


def test_should_raise_prediction_error() -> None:
    table = Table({"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(
        PredictionError,
        match=r"Error occurred while predicting: Raise ValueError \(PredictionError\) in predict for Test",
    ):
        predict(MLModelRaiseValueErrorOnFitAndPredict(), table, ["col1", "col2"], "col3")
