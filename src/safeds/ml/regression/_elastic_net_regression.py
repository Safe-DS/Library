from typing import Optional

# noinspection PyProtectedMember
import safeds._util._util_sklearn
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from sklearn.linear_model import ElasticNet as sk_ElasticNet


# noinspection PyProtectedMember
class ElasticNetRegression:
    """
    This class implements elastic net regression. It is used as a regression model.
    It can only be trained on a supervised dataset.
    """

    def __init__(self) -> None:
        self._regression = sk_ElasticNet()
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
            self._regression, supervised_dataset
        )

    def predict(self, dataset: Table, target_name: Optional[str] = None) -> Table:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset : Table
            The dataset containing the feature vectors.
        target_name: Optional[str]
            The name of the target vector. The name of the target column inferred from fit is used by default.

        Returns
        -------
        table : Table
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        PredictionError
            If prediction with the given dataset failed
        """
        return safeds._util._util_sklearn.predict(
            self._regression,
            dataset,
            target_name if target_name is not None else self.target_name,
        )
