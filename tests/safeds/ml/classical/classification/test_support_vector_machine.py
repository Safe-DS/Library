import sys

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical._bases._support_vector_machine_base import _Linear, _Polynomial
from safeds.ml.classical.classification import SupportVectorClassifier


def kernels() -> list[SupportVectorClassifier.Kernel]:
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
        SupportVectorClassifier.Kernel.linear(),
        SupportVectorClassifier.Kernel.sigmoid(),
        SupportVectorClassifier.Kernel.polynomial(3),
        SupportVectorClassifier.Kernel.radial_basis_function(),
    ]


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestC:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = SupportVectorClassifier(c=2).fit(training_set=training_set)
        assert fitted_model.c == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = SupportVectorClassifier(c=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.C == 2

    @pytest.mark.parametrize("c", [-1.0, 0.0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, c: float) -> None:
        with pytest.raises(OutOfBoundsError):
            SupportVectorClassifier(c=c)


class TestKernel:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        kernel = SupportVectorClassifier.Kernel.linear()
        fitted_model = SupportVectorClassifier(c=2, kernel=kernel).fit(training_set=training_set)
        assert isinstance(fitted_model.kernel, _Linear)

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        kernel = SupportVectorClassifier.Kernel.linear()
        fitted_model = SupportVectorClassifier(c=2, kernel=kernel).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert isinstance(fitted_model.kernel, _Linear)

    # def test_should_get_sklearn_arguments_linear(self) -> None:
    #     svm = SupportVectorClassifier(c=2, kernel=SupportVectorClassifier.Kernel.linear())
    #     assert isinstance(svm.kernel, _Linear)
    #     linear_kernel = svm.kernel._get_sklearn_arguments()
    #     assert linear_kernel == {
    #         "kernel": "linear",
    #     }

    @pytest.mark.parametrize("degree", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_degree_less_than_1(self, degree: int) -> None:
        with pytest.raises(OutOfBoundsError):
            SupportVectorClassifier.Kernel.polynomial(degree=degree)

    # def test_should_get_sklearn_arguments_polynomial(self) -> None:
    #     svm = SupportVectorClassifier(c=2, kernel=SupportVectorClassifier.Kernel.polynomial(degree=2))
    #     assert isinstance(svm.kernel, _Polynomial)
    #     poly_kernel = svm.kernel._get_sklearn_arguments()
    #     assert poly_kernel == {
    #         "kernel": "poly",
    #         "degree": 2,
    #     }

    def test_should_get_degree(self) -> None:
        kernel = _Polynomial(degree=3)
        assert kernel.degree == 3

    # def test_should_get_sklearn_arguments_sigmoid(self) -> None:
    #     svm = SupportVectorClassifier(c=2, kernel=SupportVectorClassifier.Kernel.sigmoid())
    #     assert isinstance(svm.kernel, _Sigmoid)
    #     sigmoid_kernel = svm.kernel._get_sklearn_arguments()
    #     assert sigmoid_kernel == {
    #         "kernel": "sigmoid",
    #     }
    #
    # def test_should_get_sklearn_arguments_rbf(self) -> None:
    #     svm = SupportVectorClassifier(c=2, kernel=SupportVectorClassifier.Kernel.radial_basis_function())
    #     assert isinstance(svm.kernel, _RadialBasisFunction)
    #     rbf_kernel = svm.kernel._get_sklearn_arguments()
    #     assert rbf_kernel == {
    #         "kernel": "rbf",
    #     }

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_kernel(
        self,
        kernel1: SupportVectorClassifier.Kernel,
        kernel2: SupportVectorClassifier.Kernel,
    ) -> None:
        assert hash(kernel1) == hash(kernel2)

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_kernel(
        self,
        kernel1: SupportVectorClassifier.Kernel,
        kernel2: SupportVectorClassifier.Kernel,
    ) -> None:
        assert hash(kernel1) != hash(kernel2)

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_equal_kernel(
        self,
        kernel1: SupportVectorClassifier.Kernel,
        kernel2: SupportVectorClassifier.Kernel,
    ) -> None:
        assert kernel1 == kernel2

    @pytest.mark.parametrize(
        ("kernel1", "kernel2"),
        ([(x, y) for x in kernels() for y in kernels() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_unequal_kernel(
        self,
        kernel1: SupportVectorClassifier.Kernel,
        kernel2: SupportVectorClassifier.Kernel,
    ) -> None:
        assert kernel1 != kernel2

    @pytest.mark.parametrize(
        "kernel",
        ([SupportVectorClassifier.Kernel.polynomial(3)]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_sizeof_kernel(
        self,
        kernel: SupportVectorClassifier.Kernel,
    ) -> None:
        assert sys.getsizeof(kernel) > sys.getsizeof(object())
