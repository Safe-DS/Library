from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _get_random_seed, _structural_hash

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class LogisticClassifier(Classifier):
    """Regularized logistic regression for classification."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> LogisticClassifier:
        return LogisticClassifier()

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

        return SklearnLogisticRegression(
            random_state=_get_random_seed(),
            n_jobs=-1,
        )
