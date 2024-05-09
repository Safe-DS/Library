from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError
from safeds.ml.classical._util_sklearn import fit, predict
from safeds.ml.classical.regression import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.svm import SVC as sk_SVR  # noqa: N811

    from safeds.data.labeled.containers import ExperimentalTabularDataset, TabularDataset
    from safeds.data.tabular.containers import ExperimentalTable, Table


class SupportVectorMachineKernel(ABC):
    """The abstract base class of the different subclasses supported by the `Kernel`."""

    @abstractmethod
    def _get_sklearn_arguments(self) -> dict[str, Any]:  # TODO: use apply pattern (imputer strategy) instead
        """Return the arguments to pass to scikit-learn."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Compare two kernels.

        Parameters
        ----------
        other:
            other object to compare to

        Returns
        -------
        equals:
            Whether the two kernels are equal
        """

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this kernel.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.__class__.__qualname__)


class SupportVectorMachineRegressor(Regressor):
    """
    Support vector machine.

    Parameters
    ----------
    c:
        The strength of regularization. Must be strictly positive.
    kernel:
        The type of kernel to be used. Defaults to None.

    Raises
    ------
    OutOfBoundsError
        If `c` is less than or equal to 0.
    """

    def __hash__(self) -> int:
        return _structural_hash(Regressor.__hash__(self), self._target_name, self._feature_names, self._c, self.kernel)

    def __init__(self, *, c: float = 1.0, kernel: SupportVectorMachineKernel | None = None) -> None:
        # Inputs
        if c <= 0:
            raise OutOfBoundsError(c, name="c", lower_bound=OpenBound(0))
        if kernel is None:
            kernel = self.Kernel.RadialBasisFunction()

        # Internal state
        self._wrapped_regressor: sk_SVR | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

        # Hyperparameters
        self._c: float = c
        self._kernel: SupportVectorMachineKernel = kernel

    @property
    def c(self) -> float:
        """
        Get the regularization strength.

        Returns
        -------
        result:
            The regularization strength.
        """
        return self._c

    @property
    def kernel(self) -> SupportVectorMachineKernel:
        """
        Get the type of kernel used.

        Returns
        -------
        result:
            The type of kernel used.
        """
        return self._kernel

    class Kernel:
        class Linear(SupportVectorMachineKernel):
            def _get_sklearn_arguments(self) -> dict[str, Any]:
                return {
                    "kernel": "linear",
                }

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, SupportVectorMachineRegressor.Kernel.Linear):
                    return NotImplemented
                return True

            __hash__ = SupportVectorMachineKernel.__hash__

        class Polynomial(SupportVectorMachineKernel):
            def __init__(self, degree: int):
                if degree < 1:
                    raise OutOfBoundsError(degree, name="degree", lower_bound=ClosedBound(1))
                self._degree = degree

            @property
            def degree(self) -> int:
                """The degree of the polynomial kernel."""
                return self._degree

            def _get_sklearn_arguments(self) -> dict[str, Any]:
                return {
                    "kernel": "poly",
                    "degree": self._degree,
                }

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, SupportVectorMachineRegressor.Kernel.Polynomial):
                    return NotImplemented
                return self._degree == other._degree

            def __hash__(self) -> int:
                return _structural_hash(SupportVectorMachineKernel.__hash__(self), self._degree)

            def __sizeof__(self) -> int:
                """
                Return the complete size of this object.

                Returns
                -------
                size:
                    Size of this object in bytes.
                """
                return sys.getsizeof(self._degree)

        class Sigmoid(SupportVectorMachineKernel):
            def _get_sklearn_arguments(self) -> dict[str, Any]:
                return {
                    "kernel": "sigmoid",
                }

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, SupportVectorMachineRegressor.Kernel.Sigmoid):
                    return NotImplemented
                return True

            __hash__ = SupportVectorMachineKernel.__hash__

        class RadialBasisFunction(SupportVectorMachineKernel):
            def _get_sklearn_arguments(self) -> dict[str, Any]:
                return {
                    "kernel": "rbf",
                }

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, SupportVectorMachineRegressor.Kernel.RadialBasisFunction):
                    return NotImplemented
                return True

            __hash__ = SupportVectorMachineKernel.__hash__

    def fit(self, training_set: TabularDataset | ExperimentalTabularDataset) -> SupportVectorMachineRegressor:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor:
            The fitted regressor.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        TypeError
            If a table is passed instead of a tabular dataset.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = SupportVectorMachineRegressor(c=self._c, kernel=self._kernel)
        result._wrapped_regressor = wrapped_regressor
        result._feature_names = training_set.features.column_names
        result._target_name = training_set.target.name

        return result

    def predict(self, dataset: Table | ExperimentalTable | ExperimentalTabularDataset) -> TabularDataset:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset:
            The dataset containing the feature vectors.

        Returns
        -------
        table:
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
        NonNumericColumnError
            If the dataset contains non-numerical values.
        MissingValuesColumnError
            If the dataset contains missing values.
        DatasetMissesDataError
            If the dataset contains no rows.
        """
        return predict(self._wrapped_regressor, dataset, self._feature_names, self._target_name)

    @property
    def is_fitted(self) -> bool:
        """Whether the regressor is fitted."""
        return self._wrapped_regressor is not None

    def _get_sklearn_regressor(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_regressor:
            The sklearn Regressor.
        """
        from sklearn.svm import SVC as sk_SVR  # noqa: N811

        return sk_SVR(C=self._c, **(self._kernel._get_sklearn_arguments()))
