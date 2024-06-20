from __future__ import annotations

from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.ml.classical._bases import _DecisionTreeBase

from ._regressor import Regressor
from safeds.ml.hyperparameters import Choice

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from safeds.data.labeled.containers import TabularDataset


class DecisionTreeRegressor(Regressor, _DecisionTreeBase):
    """
    Decision tree regression.

    Parameters
    ----------
    max_depth:
        The maximum depth of each tree. If None, the depth is not limited. Has to be greater than 0.
    min_sample_count_in_leaves:
        The minimum number of samples that must remain in the leaves of each tree. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `max_depth` is less than 1.
    OutOfBoundsError
        If `min_sample_count_in_leaves` is less than 1.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        max_depth: int | Choice[int] | None = None,
        min_sample_count_in_leaves: int | Choice[int] = 5,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _DecisionTreeBase.__init__(
            self,
            max_depth=max_depth,
            min_sample_count_in_leaves=min_sample_count_in_leaves,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _DecisionTreeBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_depth=self._max_depth,
            min_sample_count_in_leaves=self._min_sample_count_in_leaves,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

        return SklearnDecisionTreeRegressor(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_sample_count_in_leaves,
        )

    def _check_additional_fit_preconditions(self, training_set: TabularDataset) -> None:
        if isinstance(self._max_depth, Choice) or isinstance(self._min_sample_count_in_leaves, Choice):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self, training_set: TabularDataset) -> None:
        if not isinstance(self._max_depth, Choice) and not isinstance(self._min_sample_count_in_leaves, Choice):
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[Self]:
        max_depth_choices = self._max_depth if isinstance(self._max_depth, Choice) else [
            self._max_depth]
        min_sample_count_choices = self._min_sample_count_in_leaves if isinstance(self._min_sample_count_in_leaves, Choice) else [
            self._min_sample_count_in_leaves]

        models = []
        for md in max_depth_choices:
            for msc in min_sample_count_choices:
                models.append(DecisionTreeRegressor(max_depth=md, min_sample_count_in_leaves=msc))
        return models
