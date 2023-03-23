from abc import ABC, abstractmethod
from typing import Optional

from safeds.data.tabular.containers import Table, TaggedTable


class Classifier(ABC):
    @abstractmethod
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

    @abstractmethod
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
