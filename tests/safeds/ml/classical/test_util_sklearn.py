from __future__ import annotations

import warnings

from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import SupportVectorMachine


def test_predict_should_not_warn_about_feature_names() -> None:
    """See https://github.com/Safe-DS/Stdlib/issues/154."""
    training_set = Table({"a": [1, 2, 3], "b": [2, 4, 6]}).tag_columns(target_name="b")

    model = SupportVectorMachine()
    fitted_model = model.fit(training_set)

    test_set = Table({"a": [4, 5, 6]})

    # No warning should be emitted
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X has feature names")
        fitted_model.predict(test_set)
