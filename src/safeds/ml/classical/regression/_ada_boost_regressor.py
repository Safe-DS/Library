from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.ml.classical._bases import _AdaBoostBase
from safeds.ml.hyperparameters import Choice

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
        learner: Regressor | None | Choice[Regressor | None] = None,
        max_learner_count: int | Choice[int] = 50,
        learning_rate: float | Choice[float] = 1.0,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _AdaBoostBase.__init__(
            self,
            max_learner_count=max_learner_count,
            learning_rate=learning_rate,
        )

        # Hyperparameters
        self._learner: Regressor | None | Choice[Regressor | None] = learner

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
    def learner(self) -> Regressor | None | Choice[Regressor | None]:  # type: ignore[override]
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

        assert not isinstance(self.learner, Choice)
        learner = self.learner._get_sklearn_model() if self.learner is not None else None
        return SklearnAdaBoostRegressor(
            estimator=learner,
            n_estimators=self._max_learner_count,
            learning_rate=self._learning_rate,
        )

    def _check_additional_fit_preconditions(self) -> None:
        if (
            isinstance(self._max_learner_count, Choice)
            or isinstance(self._learning_rate, Choice)
            or isinstance(self._learner, Choice)
        ):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:
        if (
            not isinstance(self._max_learner_count, Choice)
            and not isinstance(self._learning_rate, Choice)
            and not isinstance(self._learner, Choice)
        ):
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[AdaBoostRegressor]:
        learner_choices = self._learner if isinstance(self._learner, Choice) else [self._learner]
        max_learner_count_choices = (
            self._max_learner_count if isinstance(self._max_learner_count, Choice) else [self._max_learner_count]
        )
        learning_rate_choices = (
            self._learning_rate if isinstance(self._learning_rate, Choice) else [self._learning_rate]
        )

        models = []
        for learner in learner_choices:
            for mlc in max_learner_count_choices:
                for lr in learning_rate_choices:
                    models.append(AdaBoostRegressor(learner=learner, max_learner_count=mlc, learning_rate=lr))
        return models
