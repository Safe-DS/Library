import warnings

import pytest
from sklearn.ensemble import AdaBoostRegressor as sk_AdaBoostRegressor

from safeds.data.tabular.containers import Table
from safeds.ml.classical import _util_sklearn
from safeds.ml.classical.regression import LinearRegression
from safeds.ml.exceptions import UntaggedTableError


def test_predict_should_not_warn_about_feature_names() -> None:
    """See https://github.com/Safe-DS/Stdlib/issues/51."""
    training_set = Table({"a": [1, 2, 3], "b": [2, 4, 6]}).tag_columns(
        target_name="b")

    model = LinearRegression()
    fitted_model = model.fit(training_set)

    test_set = Table({"a": [4, 5, 6]})

    # No warning should be emitted
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X has feature names")
        fitted_model.predict(test_set)


def test_fit_raises_untagged_table_error() -> None:
    table = Table.from_dict(
        {
            "a": [1.0, 0.0, 0.0, 0.0],
            "b": [0.0, 1.0, 1.0, 0.0],
            "c": [0.0, 0.0, 0.0, 1.0],
        },
    )

    with pytest.raises(UntaggedTableError):
        _util_sklearn.fit(sk_AdaBoostRegressor(), table)  # type: ignore[arg-type]
