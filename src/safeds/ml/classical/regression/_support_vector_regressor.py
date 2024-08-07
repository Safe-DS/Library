from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.ml.classical._bases import _SupportVectorMachineBase
from safeds.ml.classical.regression import Regressor
from safeds.ml.hyperparameters import Choice

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class SupportVectorRegressor(Regressor, _SupportVectorMachineBase):
    """
    Support vector machine for regression.

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
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        c: float | Choice[float] = 1.0,
        kernel: SupportVectorRegressor.Kernel | None | Choice[SupportVectorRegressor.Kernel | None] = None,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _SupportVectorMachineBase.__init__(
            self,
            c=c,
            kernel=kernel,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _SupportVectorMachineBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def kernel(self) -> SupportVectorRegressor.Kernel | Choice[SupportVectorRegressor.Kernel | None]:
        """The type of kernel used."""
        return self._kernel

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> SupportVectorRegressor:
        return SupportVectorRegressor(
            c=self._c,
            kernel=self._kernel,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.svm import SVR as SklearnSVR  # noqa: N811

        result = SklearnSVR(
            C=self._c,
        )
        assert not isinstance(self._kernel, Choice)
        self._kernel._apply(result)
        return result

    def _check_additional_fit_preconditions(self) -> None:
        if isinstance(self._c, Choice) or isinstance(self.kernel, Choice):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:
        if not isinstance(self._c, Choice) and not isinstance(self.kernel, Choice):
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[SupportVectorRegressor]:
        # assert isinstance(self._c, Choice)  # this is always true and just here for linting
        c_choices = self._c if isinstance(self._c, Choice) else [self._c]
        kernel_choices = self.kernel if isinstance(self.kernel, Choice) else [self.kernel]

        models = []
        for c in c_choices:
            for kernel in kernel_choices:
                models.append(SupportVectorRegressor(c=c, kernel=kernel))
        return models
