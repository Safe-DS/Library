from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _get_random_seed, _structural_hash
from safeds.ml.classical._bases import _SupportVectorMachineBase
from safeds.ml.classical.classification import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


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
        c: float = 1.0,
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
