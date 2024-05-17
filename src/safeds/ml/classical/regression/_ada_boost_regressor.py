from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _AdaBoostBase

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class AdaBoostRegressor(Regressor, _AdaBoostBase):
    """
    Ada Boost regression.

    Parameters
    ----------
    learner:
        The learner from which the boosted ensemble is built.
    max_learner_count:
        The maximum number of learners at which boosting is terminated. In case of perfect fit, the learning procedure
        is stopped early. Has to be greater than 0.
    learning_rate:
        Weight applied to each regressor at each boosting iteration. A higher learning rate increases the contribution
        of each regressor. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `max_learner_count` or `learning_rate` are less than or equal to 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        learner: Regressor | None = None,
        max_learner_count: int = 50,
        learning_rate: float = 1.0,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _AdaBoostBase.__init__(
            self,
            max_learner_count=max_learner_count,
            learning_rate=learning_rate,
        )

        # Hyperparameters
        self._learner: Regressor | None = learner

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _AdaBoostBase.__hash__(self),
            self._learner,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def learner(self) -> Regressor | None:
        """The base learner used for training the ensemble."""
        return self._learner

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> AdaBoostRegressor:
        return AdaBoostRegressor(
            learner=self.learner,
            max_learner_count=self._max_learner_count,
            learning_rate=self._learning_rate,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.ensemble import AdaBoostRegressor as SklearnAdaBoostRegressor

        learner = self.learner._get_sklearn_model() if self.learner is not None else None
        return SklearnAdaBoostRegressor(
            estimator=learner,
            n_estimators=self._max_learner_count,
            learning_rate=self._learning_rate,
        )
