from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound, _OpenBound
from safeds.ml.hyperparameters import Choice

if TYPE_CHECKING:
    from sklearn.svm import SVC as SklearnSVC  # noqa: N811
    from sklearn.svm import SVR as SklearnSVR  # noqa: N811


class _SupportVectorMachineBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Inner classes
    # ------------------------------------------------------------------------------------------------------------------

    class Kernel(ABC):
        """
        Possible kernels for the support vector machine.

        Use the static factory methods to create instances of this class.
        """

        @abstractmethod
        def __eq__(self, other: object) -> bool: ...

        @abstractmethod
        def __hash__(self) -> int: ...

        @abstractmethod
        def __str__(self) -> str: ...

        @abstractmethod
        def _apply(self, model: SklearnSVC | SklearnSVR) -> None:
            """Set the kernel of the given model."""

        @staticmethod
        def linear() -> _SupportVectorMachineBase.Kernel:
            """Create a linear kernel."""
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def polynomial(degree: int) -> _SupportVectorMachineBase.Kernel:
            """
            Create a polynomial kernel.

            Parameters
            ----------
            degree:
                The degree of the polynomial kernel. Must be greater than 0.

            Raises
            ------
            OutOfBoundsError
                If `degree` is not greater than 0.
            """
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def radial_basis_function() -> _SupportVectorMachineBase.Kernel:
            """Create a radial basis function kernel."""
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def sigmoid() -> _SupportVectorMachineBase.Kernel:
            """Create a sigmoid kernel."""
            raise NotImplementedError  # pragma: no cover

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        c: float | Choice[float],
        kernel: _SupportVectorMachineBase.Kernel | None | Choice[_SupportVectorMachineBase.Kernel | None],
    ) -> None:
        if kernel is None:
            kernel = _SupportVectorMachineBase.Kernel.radial_basis_function()

        # Validation
        if isinstance(c, Choice):
            for value in c:
                _check_bounds("c", value, lower_bound=_OpenBound(0))
        else:
            _check_bounds("c", c, lower_bound=_OpenBound(0))

        # Hyperparameters
        self._c: float | Choice[float] = c
        self._kernel: _SupportVectorMachineBase.Kernel | Choice[_SupportVectorMachineBase.Kernel | None] = kernel

    def __hash__(self) -> int:
        return _structural_hash(
            self._c,
            self.kernel,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def c(self) -> float | Choice[float]:
        """The regularization strength."""
        return self._c

    # This property is abstract, so subclasses must declare a public return type.
    @property
    @abstractmethod
    def kernel(self) -> _SupportVectorMachineBase.Kernel | Choice[_SupportVectorMachineBase.Kernel | None]:
        """The type of kernel used."""


# ----------------------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------------------


class _Linear(_SupportVectorMachineBase.Kernel):
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

    def _apply(self, model: SklearnSVC) -> None:
        model.kernel = "linear"


class _Polynomial(_SupportVectorMachineBase.Kernel):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, degree: int):
        _check_bounds("degree", degree, lower_bound=_ClosedBound(1))

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

    def _apply(self, model: SklearnSVC) -> None:
        model.kernel = "poly"
        model.degree = self._degree


class _RadialBasisFunction(_SupportVectorMachineBase.Kernel):
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

    def _apply(self, model: SklearnSVC) -> None:
        model.kernel = "rbf"


class _Sigmoid(_SupportVectorMachineBase.Kernel):
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

    def _apply(self, model: SklearnSVC) -> None:
        model.kernel = "sigmoid"


# Override the methods with classes, so they can be used in `isinstance` calls. Unlike methods, classes define a type.
# This is needed for the DSL, where SVM kernels are variants of an enum.
_SupportVectorMachineBase.Kernel.linear = _Linear  # type: ignore[method-assign]
_SupportVectorMachineBase.Kernel.polynomial = _Polynomial  # type: ignore[method-assign]
_SupportVectorMachineBase.Kernel.radial_basis_function = _RadialBasisFunction  # type: ignore[method-assign]
_SupportVectorMachineBase.Kernel.sigmoid = _Sigmoid  # type: ignore[method-assign]
