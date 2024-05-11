from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _SupportVectorMachineBase
from safeds.ml.classical.regression import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class SupportVectorRegressor(Regressor, _SupportVectorMachineBase):
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
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        c: float = 1.0,
        kernel: SupportVectorRegressor.Kernel | None = None,
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
