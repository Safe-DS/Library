import sys

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import SupportVectorMachineClassifier
from safeds.ml.classical.classification._support_vector_machine import SupportVectorMachineKernel


def kernels() -> list[SupportVectorMachineKernel]:
    """
    Return the list of kernels to test.

    After you implemented a new kernel, add it to this list to ensure its `__hash__` and `__eq__` method work as
    expected.

    Returns
    -------
    kernels:
        The list of kernels to test.
    """
    return [
        SupportVectorMachineClassifier.Kernel.Linear(),
        SupportVectorMachineClassifier.Kernel.Sigmoid(),
        SupportVectorMachineClassifier.Kernel.Polynomial(3),
        SupportVectorMachineClassifier.Kernel.RadialBasisFunction(),
    ]


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestC:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = SupportVectorMachineClassifier(c=2).fit(training_set=training_set)
        assert fitted_model.c == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = SupportVectorMachineClassifier(c=2).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert fitted_model._wrapped_classifier.C == 2

    @pytest.mark.parametrize("c", [-1.0, 0.0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, c: float) -> None:
        with pytest.raises(OutOfBoundsError, match=rf"c \(={c}\) is not inside \(0, \u221e\)\."):
            SupportVectorMachineClassifier(c=c)


class TestKernel:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        kernel = SupportVectorMachineClassifier.Kernel.Linear()
        fitted_model = SupportVectorMachineClassifier(c=2, kernel=kernel).fit(training_set=training_set)
        assert isinstance(fitted_model.kernel, SupportVectorMachineClassifier.Kernel.Linear)

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        kernel = SupportVectorMachineClassifier.Kernel.Linear()
        fitted_model = SupportVectorMachineClassifier(c=2, kernel=kernel).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert isinstance(fitted_model.kernel, SupportVectorMachineClassifier.Kernel.Linear)

    def test_should_get_sklearn_arguments_linear(self) -> None:
        svm = SupportVectorMachineClassifier(c=2, kernel=SupportVectorMachineClassifier.Kernel.Linear())
        assert isinstance(svm.kernel, SupportVectorMachineClassifier.Kernel.Linear)
        linear_kernel = svm.kernel._get_sklearn_arguments()
        assert linear_kernel == {
            "kernel": "linear",
        }

    @pytest.mark.parametrize("degree", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_degree_less_than_1(self, degree: int) -> None:
        with pytest.raises(OutOfBoundsError, match=rf"degree \(={degree}\) is not inside \[1, \u221e\)\."):
            SupportVectorMachineClassifier.Kernel.Polynomial(degree=degree)

    def test_should_get_sklearn_arguments_polynomial(self) -> None:
        svm = SupportVectorMachineClassifier(c=2, kernel=SupportVectorMachineClassifier.Kernel.Polynomial(degree=2))
        assert isinstance(svm.kernel, SupportVectorMachineClassifier.Kernel.Polynomial)
        poly_kernel = svm.kernel._get_sklearn_arguments()
        assert poly_kernel == {
            "kernel": "poly",
            "degree": 2,
        }

    def test_should_get_degree(self) -> None:
        kernel = SupportVectorMachineClassifier.Kernel.Polynomial(degree=3)
        assert kernel.degree == 3

    def test_should_get_sklearn_arguments_sigmoid(self) -> None:
        svm = SupportVectorMachineClassifier(c=2, kernel=SupportVectorMachineClassifier.Kernel.Sigmoid())
        assert isinstance(svm.kernel, SupportVectorMachineClassifier.Kernel.Sigmoid)
        sigmoid_kernel = svm.kernel._get_sklearn_arguments()
        assert sigmoid_kernel == {
            "kernel": "sigmoid",
        }

    def test_should_get_sklearn_arguments_rbf(self) -> None:
        svm = SupportVectorMachineClassifier(c=2, kernel=SupportVectorMachineClassifier.Kernel.RadialBasisFunction())
        assert isinstance(svm.kernel, SupportVectorMachineClassifier.Kernel.RadialBasisFunction)
        rbf_kernel = svm.kernel._get_sklearn_arguments()
        assert rbf_kernel == {
            "kernel": "rbf",
        }

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_kernel(
        self,
        kernel1: SupportVectorMachineKernel,
        kernel2: SupportVectorMachineKernel,
    ) -> None:
        assert hash(kernel1) == hash(kernel2)

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_kernel(
        self,
        kernel1: SupportVectorMachineKernel,
        kernel2: SupportVectorMachineKernel,
    ) -> None:
        assert hash(kernel1) != hash(kernel2)

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_equal_kernel(
        self,
        kernel1: SupportVectorMachineKernel,
        kernel2: SupportVectorMachineKernel,
    ) -> None:
        assert kernel1 == kernel2

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_unequal_kernel(
        self,
        kernel1: SupportVectorMachineKernel,
        kernel2: SupportVectorMachineKernel,
    ) -> None:
        assert kernel1 != kernel2

    @pytest.mark.parametrize(
        "kernel",
        ([SupportVectorMachineClassifier.Kernel.Polynomial(3)]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_sizeof_kernel(
        self,
        kernel: SupportVectorMachineKernel,
    ) -> None:
        assert sys.getsizeof(kernel) > sys.getsizeof(object())
