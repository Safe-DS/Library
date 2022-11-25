from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import LearningError, PredictionError
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression


# noinspection PyProtectedMember
class LogisticRegression:
    """
    This class implements regularized logistic regression. It is used as a classifier model.
    It can only be trained on a supervised dataset.
    """

    def __init__(self):
        self._clf = sk_LogisticRegression(n_jobs=-1)

    def fit(self, supervised_dataset: SupervisedDataset):
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
        try:
            self._clf.fit(
                supervised_dataset.feature_vectors._data,
                supervised_dataset.target_values._data,
            )
        except ValueError as exception:
            raise LearningError(str(exception)) from exception
        except Exception as exception:
            raise LearningError(None) from exception

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
        try:
            predicted_target_vector = self._clf.predict(dataset._data)
            result_set = dataset._data.copy(deep=True)
            if "target_predictions" in result_set.columns:
                raise ValueError(
                    "Dataset already contains 'target_predictions' column. Please rename this column"
                )
            result_set["target_predictions"] = predicted_target_vector
            return Table(result_set)
        except NotFittedError as exception:
            raise PredictionError("The model was not trained") from exception
        except ValueError as exception:
            raise PredictionError(str(exception)) from exception
        except Exception as exception:
            raise PredictionError(None) from exception
