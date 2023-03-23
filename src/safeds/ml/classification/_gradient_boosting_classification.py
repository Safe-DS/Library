from typing import Optional

# noinspection PyProtectedMember
import safeds.ml._util_sklearn
from safeds.data.tabular.containers import Table, TaggedTable
from sklearn.ensemble import GradientBoostingClassifier

from ._classifier import Classifier


# noinspection PyProtectedMember
class GradientBoosting(Classifier):
    """
    This class implements gradient boosting classification. It is used as a classifier model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._classification = GradientBoostingClassifier()
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
        self.target_name = safeds.ml._util_sklearn.fit(
            self._classification, tagged_table
        )

    # noinspection PyProtectedMember
    def predict(self, dataset: Table, target_name: Optional[str] = None) -> Table:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset : Table
            The dataset containing the feature vectors.
        target_name : Optional[str]
            The name of the target vector. The name of the target column inferred from fit is used by default.

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
            self._classification,
            dataset,
            target_name if target_name is not None else self.target_name,
        )
