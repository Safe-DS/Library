from safeds.data.labeled.containers import TabularDataset
from safeds.ml.nn._converters import (
    _TableConverter,
)


def test_should_raise_if_is_fitted_is_set_correctly_lstm() -> None:
    it = _TableConverter()
    it._feature_names = ["b"]
    assert it._is_fit_data_valid(TabularDataset({"a": [1], "b": [1]}, "a"))
