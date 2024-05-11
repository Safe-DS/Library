from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class AdaBoostClassifier(Classifier):
    """
    Ada Boost classification.

    Parameters
    ----------
    learner:
        The learner from which the boosted ensemble is built.
    maximum_number_of_learners:
        The maximum number of learners at which boosting is terminated. In case of perfect fit, the learning procedure
        is stopped early. Has to be greater than 0.
    learning_rate:
        Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution
        of each classifier. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `maximum_number_of_learners` or `learning_rate` are less than or equal to 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        learner: Classifier | None = None,
        maximum_number_of_learners: int = 50,
        learning_rate: float = 1.0,
    ) -> None:
        super().__init__()

        # Validation
        if maximum_number_of_learners < 1:
            raise OutOfBoundsError(
                maximum_number_of_learners,
                name="maximum_number_of_learners",
                lower_bound=ClosedBound(1),
            )
        if learning_rate <= 0:
            raise OutOfBoundsError(learning_rate, name="learning_rate", lower_bound=OpenBound(0))

        # Hyperparameters
        self._learner = learner
        self._maximum_number_of_learners = maximum_number_of_learners
        self._learning_rate = learning_rate

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._maximum_number_of_learners,
            self._learning_rate,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def learner(self) -> Classifier | None:
        """The base learner used for training the ensemble."""
        return self._learner

    @property
    def maximum_number_of_learners(self) -> int:
        """The maximum number of learners in the ensemble."""
        return self._maximum_number_of_learners

    @property
    def learning_rate(self) -> float:
        """The learning rate."""
        return self._learning_rate

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> AdaBoostClassifier:
        return AdaBoostClassifier(
            learner=self.learner,
            maximum_number_of_learners=self.maximum_number_of_learners,
            learning_rate=self._learning_rate,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier

        learner = self.learner._get_sklearn_model() if self.learner is not None else None
        return SklearnAdaBoostClassifier(
            estimator=learner,
            n_estimators=self.maximum_number_of_learners,
            learning_rate=self._learning_rate,
        )
