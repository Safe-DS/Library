from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _get_random_seed, _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError
from safeds.ml.classical.classification import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin
    from sklearn.svm import SVC as SklearnSVM  # noqa: N811

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class SupportVectorClassifier(Classifier):
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
        def _apply(self, model: SklearnSVM) -> None:
            """Set the kernel of the given model."""

        @staticmethod
        def Linear() -> SupportVectorClassifier.Kernel:  # noqa: N802
            """A linear kernel."""  # noqa: D401
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def Polynomial(degree: int) -> SupportVectorClassifier.Kernel:  # noqa: N802
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
        def RadialBasisFunction() -> SupportVectorClassifier.Kernel:  # noqa: N802
            """A radial basis function kernel."""  # noqa: D401
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def Sigmoid() -> SupportVectorClassifier.Kernel:  # noqa: N802
            """A sigmoid kernel."""  # noqa: D401
            raise NotImplementedError  # pragma: no cover

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        c: float = 1.0,
        kernel: SupportVectorClassifier.Kernel | None = None,
    ) -> None:
        super().__init__()

        # Inputs
        if c <= 0:
            raise OutOfBoundsError(c, name="c", lower_bound=OpenBound(0))
        if kernel is None:
            kernel = SupportVectorClassifier.Kernel.RadialBasisFunction()

        # Hyperparameters
        self._c: float = c
        self._kernel: SupportVectorClassifier.Kernel = kernel

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
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
    def kernel(self) -> SupportVectorClassifier.Kernel:
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

    def _clone(self) -> SupportVectorClassifier:
        return SupportVectorClassifier(
            c=self._c,
            kernel=self._kernel,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier:
            The sklearn Classifier.
        """
        from sklearn.svm import SVC as SklearnSVC  # noqa: N811

        result = SklearnSVC(
            C=self._c,
            random_state=_get_random_seed(),
        )
        self._kernel._apply(result)
        return result


# ----------------------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------------------

class _Linear(SupportVectorClassifier.Kernel):

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

    def _apply(self, model: SklearnSVM) -> None:
        model.kernel = "linear"


class _Polynomial(SupportVectorClassifier.Kernel):

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

    def _apply(self, model: SklearnSVM) -> None:
        model.kernel = "poly"
        model.degree = self._degree


class _RadialBasisFunction(SupportVectorClassifier.Kernel):

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

    def _apply(self, model: SklearnSVM) -> None:
        model.kernel = "rbf"


class _Sigmoid(SupportVectorClassifier.Kernel):

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

    def _apply(self, model: SklearnSVM) -> None:
        model.kernel = "sigmoid"


# Override the methods with classes, so they can be used in `isinstance` calls. Unlike methods, classes define a type.
# This is needed for the DSL, where SVM kernels are variants of an enum.
SupportVectorClassifier.Kernel.Linear = _Linear  # type: ignore[method-assign]
SupportVectorClassifier.Kernel.Polynomial = _Polynomial  # type: ignore[method-assign]
SupportVectorClassifier.Kernel.RadialBasisFunction = _RadialBasisFunction  # type: ignore[method-assign]
SupportVectorClassifier.Kernel.Sigmoid = _Sigmoid  # type: ignore[method-assign]
