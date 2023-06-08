import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.regression._support_vector_machine import SupportVectorMachine


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestC:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        kernel = SupportVectorMachine.Kernel.Linear()
        fitted_model = SupportVectorMachine(c=2, kernel=kernel).fit(training_set=training_set)
        assert fitted_model._c == 2
        assert isinstance(fitted_model.kernel, SupportVectorMachine.Kernel.Linear)

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        kernel = SupportVectorMachine.Kernel.Linear()
        fitted_model = SupportVectorMachine(c=2, kernel=kernel).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.C == 2
        assert isinstance(fitted_model.kernel, SupportVectorMachine.Kernel.Linear)

    def test_should_raise_if_less_than_or_equal_to_0(self) -> None:
        with pytest.raises(
            ValueError,
            match="The parameter 'c' has to be strictly positive.",
        ):
            SupportVectorMachine(c=-1)
