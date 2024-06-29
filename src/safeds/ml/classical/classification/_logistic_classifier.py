from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _get_random_seed, _structural_hash
from safeds._validation import _check_bounds, _OpenBound

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class LogisticClassifier(Classifier):
    """
    Regularized logistic regression for classification.

    Parameters
    ----------
    c:
        The regularization strength. Lower values imply stronger regularization. Must be greater than 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, *, c: float = 1.0) -> None:
        super().__init__()

        # Validation
        _check_bounds("c", c, lower_bound=_OpenBound(0))

        # Hyperparameters
        self._c: float = c

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def c(self) -> float:
        """The regularization strength. Lower values imply stronger regularization."""
        return self._c

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> LogisticClassifier:
        return LogisticClassifier(c=self.c)

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

        return SklearnLogisticRegression(
            random_state=_get_random_seed(),
            n_jobs=-1,
            C=self.c,
        )
