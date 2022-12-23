# noinspection PyProtectedMember
import safe_ds._util._util_sklearn
from safe_ds.data import SupervisedDataset, Table
from sklearn.linear_model import Lasso as sk_LassoRegression


# noinspection PyProtectedMember
class LassoRegression:
    """
    This class implements lasso regression. It is used as a regression model.
    It can only be trained on a supervised dataset.
    """

    def __init__(self) -> None:
        self._regression = sk_LassoRegression()

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
        safe_ds._util._util_sklearn.fit(self._regression, supervised_dataset)

    def predict(self, dataset: Table) -> Table:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first

        Parameters
        ----------
        dataset: Table
            the dataset containing the feature vectors

        Returns
        -------
        table : Table
            a dataset containing the given feature vectors and the predicted target vector

        Raises
        ------
        PredictionError
            if predicting with the given dataset failed
        """
        return safe_ds._util._util_sklearn.predict(self._regression, dataset)
