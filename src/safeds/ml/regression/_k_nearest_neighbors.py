# noinspection PyProtectedMember
from safeds.ml._util_sklearn import fit, predict
from safeds.data.tabular.containers import Table, TaggedTable
from sklearn.neighbors import KNeighborsRegressor as sk_KNeighborsRegressor

from ._regressor import Regressor


# noinspection PyProtectedMember
class KNearestNeighbors(Regressor):
    """
    This class implements K-nearest-neighbors regressor. It can only be trained on a tagged table.

    Parameters
    ----------
    n_neighbors : int
        The number of neighbors to be interpolated with. Has to be less than or equal than the sample size.
    """

    def __init__(self, n_neighbors: int) -> None:
        self._wrapped_regressor = sk_KNeighborsRegressor(n_neighbors)
        self._target_name = ""

    def fit(self, training_set: TaggedTable) -> None:
        """
        Fit this model given a tagged table.

        Parameters
        ----------
        training_set : TaggedTable
            The tagged table containing the feature and target vectors.

        Raises
        ------
        LearningError
            If the tagged table contains invalid values or if the training failed.
        """
        fit(self._wrapped_regressor, training_set)
        self._target_name = training_set.target_values.name

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
