from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError
from safeds.ml.classical.regression import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.svm import SVR as SklearnSVR  # noqa: N811

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class SupportVectorRegressor(Regressor):
    """
    Support vector machine for classification.

    Parameters
    ----------
    c:
        The strength of regularization. Must be greater than 0.
    kernel:
        The type of kernel to be used. Defaults to a radial basis function kernel.

    Raises
    ------
    OutOfBoundsError
        If `c` is less than or equal to 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Inner classes
    # ------------------------------------------------------------------------------------------------------------------

    class Kernel(ABC):
        """Possible kernels for the support vector machine. Use the inner classes to create instances of this class."""

        @abstractmethod
        def __eq__(self, other: object) -> bool: ...

        @abstractmethod
        def __hash__(self) -> int: ...

        @abstractmethod
        def __str__(self) -> str: ...

        @abstractmethod
        def _apply(self, model: SklearnSVR) -> None:
            """Set the kernel of the given model."""

        @staticmethod
        def Linear() -> SupportVectorRegressor.Kernel:  # noqa: N802
            """A linear kernel."""  # noqa: D401
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def Polynomial(degree: int) -> SupportVectorRegressor.Kernel:  # noqa: N802
            """
            A polynomial kernel.

            Parameters
            ----------
            degree:
                The degree of the polynomial kernel. Must be greater than 0.

            Raises
            ------
            ValueError
                If `degree` is not greater than 0.
            """  # noqa: D401
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def RadialBasisFunction() -> SupportVectorRegressor.Kernel:  # noqa: N802
            """A radial basis function kernel."""  # noqa: D401
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def Sigmoid() -> SupportVectorRegressor.Kernel:  # noqa: N802
            """A sigmoid kernel."""  # noqa: D401
            raise NotImplementedError  # pragma: no cover

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        c: float = 1.0,
        kernel: SupportVectorRegressor.Kernel | None = None,
    ) -> None:
        super().__init__()

        # Inputs
        if c <= 0:
            raise OutOfBoundsError(c, name="c", lower_bound=OpenBound(0))
        if kernel is None:
            kernel = SupportVectorRegressor.Kernel.RadialBasisFunction()

        # Hyperparameters
        self._c: float = c
        self._kernel: SupportVectorRegressor.Kernel = kernel

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            self._c,
            self.kernel,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

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
    def kernel(self) -> SupportVectorRegressor.Kernel:
        """
        Get the type of kernel used.

        Returns
        -------
        result:
            The type of kernel used.
        """
        return self._kernel

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> SupportVectorRegressor:
        return SupportVectorRegressor(
            c=self._c,
            kernel=self._kernel,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Regressor.
        """
        from sklearn.svm import SVR as SklearnSVR  # noqa: N811

        result = SklearnSVR(
            C=self._c,
        )
        self._kernel._apply(result)
        return result


# ----------------------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------------------

class _Linear(SupportVectorRegressor.Kernel):

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Linear):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__qualname__)

    def __str__(self) -> str:
        return "Linear"

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _apply(self, model: SklearnSVR) -> None:
        model.kernel = "linear"


class _Polynomial(SupportVectorRegressor.Kernel):

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, degree: int):
        if degree < 1:
            raise OutOfBoundsError(degree, name="degree", lower_bound=ClosedBound(1))

        self._degree = degree

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Polynomial):
            return NotImplemented
        return self._degree == other._degree

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._degree,
        )

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._degree)

    def __str__(self) -> str:
        return f"Polynomial(degree={self._degree})"

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def degree(self) -> int:
        """The degree of the polynomial kernel."""
        return self._degree

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _apply(self, model: SklearnSVR) -> None:
        model.kernel = "poly"
        model.degree = self._degree


class _RadialBasisFunction(SupportVectorRegressor.Kernel):

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _RadialBasisFunction):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__qualname__)

    def __str__(self) -> str:
        return "RadialBasisFunction"

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _apply(self, model: SklearnSVR) -> None:
        model.kernel = "rbf"


class _Sigmoid(SupportVectorRegressor.Kernel):

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Sigmoid):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__qualname__)

    def __str__(self) -> str:
        return "Sigmoid"

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _apply(self, model: SklearnSVR) -> None:
        model.kernel = "sigmoid"


# Override the methods with classes, so they can be used in `isinstance` calls. Unlike methods, classes define a type.
# This is needed for the DSL, where SVM kernels are variants of an enum.
SupportVectorRegressor.Kernel.Linear = _Linear  # type: ignore[method-assign]
SupportVectorRegressor.Kernel.Polynomial = _Polynomial  # type: ignore[method-assign]
SupportVectorRegressor.Kernel.RadialBasisFunction = _RadialBasisFunction  # type: ignore[method-assign]
SupportVectorRegressor.Kernel.Sigmoid = _Sigmoid  # type: ignore[method-assign]
