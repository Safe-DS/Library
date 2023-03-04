from typing import Optional

# noinspection PyProtectedMember
import safeds._util._util_sklearn
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from sklearn.neighbors import KNeighborsClassifier


# noinspection PyProtectedMember
class KNearestNeighbors:
    """
    This class implements K-nearest-neighbors classifier. It can only be trained on a supervised dataset.

    Parameters
    ----------
    n_neighbors : int
        The number of neighbors to be interpolated with. Has to be less than or equal to the sample size.
    """

    def __init__(self, n_neighbors: int) -> None:
        self._classification = KNeighborsClassifier(n_jobs=-1, n_neighbors=n_neighbors)
        self.target_name = ""

    def fit(self, supervised_dataset: SupervisedDataset) -> None:
        """
        Fit this model given a supervised dataset.

        Parameters
        ----------
        supervised_dataset : SupervisedDataset
            The supervised dataset containing the feature and target vectors.

        Raises
        ------
        LearningError
            If the supervised dataset contains invalid values or if the training failed.
        """
        self.target_name = safeds._util._util_sklearn.fit(
            self._classification, supervised_dataset
        )

    def predict(self, dataset: Table, target_name: Optional[str] = None) -> Table:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first

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
        return safeds._util._util_sklearn.predict(
            self._classification,
            dataset,
            target_name if target_name is not None else self.target_name,
        )
