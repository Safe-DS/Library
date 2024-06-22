from __future__ import annotations

from typing import TYPE_CHECKING, Self

from safeds._utils import _get_random_seed, _structural_hash
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.ml.classical._bases import _SupportVectorMachineBase
from safeds.ml.classical.classification import Classifier
from safeds.ml.hyperparameters import Choice

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class SupportVectorClassifier(Classifier, _SupportVectorMachineBase):
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
        c: float | Choice[float] = 1.0,
        kernel: SupportVectorClassifier.Kernel | None = None,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _SupportVectorMachineBase.__init__(
            self,
            c=c,
            kernel=kernel,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _SupportVectorMachineBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def kernel(self) -> SupportVectorClassifier.Kernel:
        """The type of kernel used."""
        return self._kernel

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> SupportVectorClassifier:
        return SupportVectorClassifier(
            c=self._c,
            kernel=self._kernel,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.svm import SVC as SklearnSVC  # noqa: N811

        result = SklearnSVC(
            C=self._c,
            random_state=_get_random_seed(),
        )
        self._kernel._apply(result)
        return result

    def _check_additional_fit_preconditions(self) -> None:
        if isinstance(self._c, Choice):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:
        if not isinstance(self._c, Choice):
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[Self]:
        models = []
        for c in self._c:
            models.append(SupportVectorClassifier(c=c))
        return models
