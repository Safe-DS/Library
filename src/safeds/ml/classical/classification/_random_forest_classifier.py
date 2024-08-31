from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _get_random_seed, _structural_hash
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.ml.classical._bases import _RandomForestBase
from safeds.ml.hyperparameters import Choice

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class RandomForestClassifier(Classifier, _RandomForestBase):
    """
    Random forest classification.

    Parameters
    ----------
    tree_count:
        The number of trees to be used in the random forest. Has to be greater than 0.
    max_depth:
        The maximum depth of each tree. If None, the depth is not limited. Has to be greater than 0.
    min_sample_count_in_leaves:
        The minimum number of samples that must remain in the leaves of each tree. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `tree_count` is less than 1.
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
        tree_count: int | Choice[int] = 100,
        max_depth: int | None | Choice[int | None] = None,
        min_sample_count_in_leaves: int | Choice[int] = 1,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _RandomForestBase.__init__(
            self,
            tree_count=tree_count,
            max_depth=max_depth,
            min_sample_count_in_leaves=min_sample_count_in_leaves,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _RandomForestBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            tree_count=self._tree_count,
            max_depth=self._max_depth,
            min_sample_count_in_leaves=self._min_sample_count_in_leaves,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

        return SklearnRandomForestClassifier(
            n_estimators=self._tree_count,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_sample_count_in_leaves,
            random_state=_get_random_seed(),
            n_jobs=-1,
        )

    def _check_additional_fit_preconditions(self) -> None:
        if (
            isinstance(self._tree_count, Choice)
            or isinstance(self._max_depth, Choice)
            or isinstance(self._min_sample_count_in_leaves, Choice)
        ):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:
        if (
            not isinstance(self._tree_count, Choice)
            and not isinstance(self._max_depth, Choice)
            and not isinstance(self._min_sample_count_in_leaves, Choice)
        ):
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[RandomForestClassifier]:
        tree_count_choices = self._tree_count if isinstance(self._tree_count, Choice) else [self._tree_count]
        max_depth_choices = self._max_depth if isinstance(self._max_depth, Choice) else [self._max_depth]
        min_sample_count_choices = (
            self._min_sample_count_in_leaves
            if isinstance(self._min_sample_count_in_leaves, Choice)
            else [self._min_sample_count_in_leaves]
        )

        models = []
        for tc in tree_count_choices:
            for md in max_depth_choices:
                for msc in min_sample_count_choices:
                    models.append(RandomForestClassifier(tree_count=tc, max_depth=md, min_sample_count_in_leaves=msc))
        return models
