from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError
from safeds.ml.classical._bases import _GradientBoostingBase

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class GradientBoostingClassifier(Classifier, _GradientBoostingBase):
    """
    Gradient boosting classification.

    Parameters
    ----------
    number_of_trees:
        The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large
        number usually results in better performance.
    learning_rate:
        The larger the value, the more the model is influenced by each additional tree. If the learning rate is too
        low, the model might underfit. If the learning rate is too high, the model might overfit.

    Raises
    ------
    OutOfBoundsError
        If `number_of_trees` or `learning_rate` is less than or equal to 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        number_of_trees: int = 100,
        learning_rate: float = 0.1,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _GradientBoostingBase.__init__(
            self,
            number_of_trees=number_of_trees,
            learning_rate=learning_rate,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _GradientBoostingBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            number_of_trees=self._number_of_trees,
            learning_rate=self._learning_rate,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier

        return SklearnGradientBoostingClassifier(
            n_estimators=self._number_of_trees,
            learning_rate=self._learning_rate,
        )
