# noinspection PyProtectedMember
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml._util_sklearn import fit, predict
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from ._classifier import Classifier


# noinspection PyProtectedMember
class LogisticRegression(Classifier):
    """
    This class implements regularized logistic regression. It is used as a classifier model.
    It can only be trained on a tagged table.
    """

    def __init__(self) -> None:
        self._wrapped_classifier = sk_LogisticRegression(n_jobs=-1)
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
        fit(self._wrapped_classifier, training_set)
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
        return predict(self._wrapped_classifier, dataset, self._target_name)
