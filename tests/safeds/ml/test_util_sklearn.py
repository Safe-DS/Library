import warnings

from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.ml.regression import LinearRegression


def test_predict_should_not_warn_about_feature_names() -> None:
    """
    See https://github.com/Safe-DS/Stdlib/issues/51.
    """

    training_set = SupervisedDataset(
        Table({"a": [1, 2, 3], "b": [2, 4, 6]}), target_column="b"
    )

    model = LinearRegression()
    model.fit(training_set)

    test_set = Table({"a": [4, 5, 6]})

    # No warning should be emitted
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X has feature names")
        model.predict(dataset=test_set)
