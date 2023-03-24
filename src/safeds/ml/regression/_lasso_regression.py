from typing import Optional

# noinspection PyProtectedMember
import safeds.ml._util_sklearn
from safeds.data.tabular.containers import Table, TaggedTable
from sklearn.linear_model import Lasso as sk_Lasso

from ._regressor import Regressor


# noinspection PyProtectedMember
class LassoRegression(Regressor):
    """
    This class implements lasso regression. It is used as a regression model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._regression = sk_Lasso()
        self.target_name = ""

    def fit(self, tagged_table: TaggedTable) -> None:
        """
        Fit this model given a tagged table.

        Parameters
        ----------
        tagged_table : TaggedTable
            The tagged table containing the feature and target vectors.

        Raises
        ------
        LearningError
            If the tagged table contains invalid values or if the training failed.
        """
        self.target_name = safeds.ml._util_sklearn.fit(self._regression, tagged_table)

    def predict(self, dataset: Table) -> Table:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset : Table
            The dataset containing the feature vectors.

        Returns
        -------
        table : Table
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        PredictionError
            If prediction with the given dataset failed.
        """
        return safeds.ml._util_sklearn.predict(
            self._regression,
            dataset,
            self.target_name,
        )
