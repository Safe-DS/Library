from typing import Optional

# noinspection PyProtectedMember
import safe_ds._util._util_sklearn
from safe_ds.data import SupervisedDataset, Table
from sklearn.ensemble import RandomForestRegressor


# noinspection PyProtectedMember
class RandomForest:
    """
    This class implements Random Forest regression. It can only be trained on a supervised dataset.
    """

    def __init__(self) -> None:
        self._regression = RandomForestRegressor(n_jobs=-1)
        self.target_name = ""

    def fit(self, supervised_dataset: SupervisedDataset) -> None:
        """
        Fit this model given a supervised dataset.

        Parameters
        ----------
        supervised_dataset: SupervisedDataset
            the supervised dataset containing the feature and target vectors

        Raises
        ------
        LearningError
            if the supervised dataset contains invalid values or if the training failed
        """
        self.target_name = safe_ds._util._util_sklearn.fit(
            self._regression, supervised_dataset
        )

    def predict(self, dataset: Table, target_name: Optional[str] = None) -> Table:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first

        Parameters
        ----------
        dataset: Table
            the dataset containing the feature vectors
        target_name: Optional[str]
            the name of the target vector, the name of the target column inferred from fit is used by default

        Returns
        -------
        table : Table
            a dataset containing the given feature vectors and the predicted target vector

        Raises
        ------
        PredictionError
            if predicting with the given dataset failed
        """
        return safe_ds._util._util_sklearn.predict(
            self._regression,
            dataset,
            target_name if target_name is not None else self.target_name,
        )
