from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.image.containers import Image
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.exceptions._ml import ModelNotFittedError
from safeds.ml.classical._bases import _DecisionTreeBase
from safeds.ml.hyperparameters import Choice

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class DecisionTreeClassifier(Classifier, _DecisionTreeBase):
    """
    Decision tree classification.

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
        max_depth: int | None | Choice[int | None] = None,
        min_sample_count_in_leaves: int | Choice[int] = 1,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _DecisionTreeBase.__init__(
            self,
            max_depth=max_depth,
            min_sample_count_in_leaves=min_sample_count_in_leaves,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _DecisionTreeBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self._max_depth,
            min_sample_count_in_leaves=self._min_sample_count_in_leaves,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

        return SklearnDecisionTreeClassifier(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_sample_count_in_leaves,
        )

    def _check_additional_fit_preconditions(self) -> None:
        if isinstance(self._max_depth, Choice) or isinstance(self._min_sample_count_in_leaves, Choice):
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:
        if not isinstance(self._max_depth, Choice) and not isinstance(self._min_sample_count_in_leaves, Choice):
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[DecisionTreeClassifier]:
        max_depth_choices = self._max_depth if isinstance(self._max_depth, Choice) else [self._max_depth]
        min_sample_count_choices = (
            self._min_sample_count_in_leaves
            if isinstance(self._min_sample_count_in_leaves, Choice)
            else [self._min_sample_count_in_leaves]
        )

        models = []
        for md in max_depth_choices:
            for msc in min_sample_count_choices:
                models.append(DecisionTreeClassifier(max_depth=md, min_sample_count_in_leaves=msc))
        return models

    # ------------------------------------------------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------------------------------------------------

    def plot(self) -> Image:
        """
        Get the image of the decision tree.

        Returns
        -------
        plot:
            The decision tree figure as an image.

        Raises
        ------
        ModelNotFittedError:
            If model is not fitted.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        from io import BytesIO

        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        plot_tree(self._wrapped_model)

        # save plot fig bytes in buffer
        with BytesIO() as buffer:
            plt.savefig(buffer)
            image = buffer.getvalue()

        # prevent forced plot from sklearn showing
        plt.close()

        return Image.from_bytes(image)
