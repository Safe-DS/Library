# noinspection PyProtectedMember
from safeds.ml._util_sklearn import fit, predict
from safeds.data.tabular.containers import Table, TaggedTable
from sklearn.ensemble import GradientBoostingRegressor as sk_GradientBoostingRegressor

from ._regressor import Regressor


# noinspection PyProtectedMember
class GradientBoosting(Regressor):
    """
    This class implements gradient boosting regression. It is used as a regression model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._wrapped_regressor = sk_GradientBoostingRegressor()
        self._target_name = ""

    def fit(self, training_set: TaggedTable) -> None:
        """
        Fit this model given a tagged table.

        Parameters

        ----------
        tagged_table : SupervisedDataset
            The tagged table containing the feature and target vectors.

        Raises
        ------
        LearningError
            If the tagged table contains invalid values or if the training failed.
        """
        fit(self._wrapped_regressor, training_set)
        self._target_name = training_set.target_values.name

        # noinspection PyProtectedMember
    def predict(self, dataset: Table) -> TaggedTable:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset : Table
            The dataset containing the feature vectors.

        Returns
        -------
        table : TaggedTable
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        PredictionError
            If prediction with the given dataset failed.
        """
        return predict(self._wrapped_regressor, dataset, self._target_name)
