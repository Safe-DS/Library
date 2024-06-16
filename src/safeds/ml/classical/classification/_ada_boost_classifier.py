from __future__ import annotations

from typing import TYPE_CHECKING, Self, Any

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError, LearningError
from safeds.ml.classical._bases import _AdaBoostBase

from ._classifier import Classifier
from ...hyperparameters import Choice
from ...metrics import ClassifierMetric

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class AdaBoostClassifier(Classifier, _AdaBoostBase):
    """
    Ada Boost classification.

    Parameters
    ----------
    learner:
        The learner from which the boosted ensemble is built.
    max_learner_count:
        The maximum number of learners at which boosting is terminated. In case of perfect fit, the learning procedure
        is stopped early. Has to be greater than 0.
    learning_rate:
        Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution
        of each classifier. Has to be greater than 0.

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
        learner: Classifier | None = None,
        max_learner_count: int | Choice[int] = 50,
        learning_rate: float = 1.0,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _AdaBoostBase.__init__(
            self,
            max_learner_count=max_learner_count,
            learning_rate=learning_rate,
        )

        # Hyperparameters
        self._learner: Classifier | None = learner

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _AdaBoostBase.__hash__(self),
            self._learner,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def learner(self) -> Classifier | None:
        """The base learner used for training the ensemble."""
        return self._learner

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> AdaBoostClassifier:
        return AdaBoostClassifier(
            learner=self.learner,
            max_learner_count=self._max_learner_count,
            learning_rate=self._learning_rate,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier
        learner = self.learner._get_sklearn_model() if self.learner is not None else None
        return SklearnAdaBoostClassifier(
            estimator=learner,
            n_estimators=self._max_learner_count,
            learning_rate=self._learning_rate,
        )

    def _check_additional_fit_preconditions(self, training_set: TabularDataset) -> None:
        if isinstance(self._max_learner_count, Choice):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self, training_set: TabularDataset) -> None:
        if isinstance(self._max_learner_count, int):
            raise FittingWithoutChoiceError


    def _get_models_for_all_choices(self) -> list[Self]:
        models = []
        for value in self._max_learner_count:
            models.append(AdaBoostClassifier(learner=self.learner, max_learner_count=value, learning_rate=self.learning_rate))
        return models
