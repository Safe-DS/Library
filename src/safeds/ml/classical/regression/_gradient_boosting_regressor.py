from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class GradientBoostingRegressor(Regressor):
    """
    Gradient boosting regression.

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
        If `number_of_trees` or `learning_rate` are less than or equal to 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, *, number_of_trees: int = 100, learning_rate: float = 0.1) -> None:
        super().__init__()

        # Validation
        if number_of_trees < 1:
            raise OutOfBoundsError(number_of_trees, name="number_of_trees", lower_bound=ClosedBound(1))
        if learning_rate <= 0:
            raise OutOfBoundsError(learning_rate, name="learning_rate", lower_bound=OpenBound(0))

        # Hyperparameters
        self._number_of_trees = number_of_trees
        self._learning_rate = learning_rate

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._number_of_trees,
            self._learning_rate,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_trees(self) -> int:
        """
        Get the number of trees (estimators) in the ensemble.

        Returns
        -------
        result:
            The number of trees.
        """
        return self._number_of_trees

    @property
    def learning_rate(self) -> float:
        """
        Get the learning rate.

        Returns
        -------
        result:
            The learning rate.
        """
        return self._learning_rate

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            number_of_trees=self._number_of_trees,
            learning_rate=self._learning_rate,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor

        return SklearnGradientBoostingRegressor(
            n_estimators=self._number_of_trees,
            learning_rate=self._learning_rate,
        )
