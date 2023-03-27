from typing import Any, Optional

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError, PredictionError
from sklearn.exceptions import NotFittedError


# noinspection PyProtectedMember
def fit(model: Any, tagged_table: TaggedTable) -> None:
    """
    Fit a model for a given tagged table.

    Parameters
    ----------
    model
        Classifier or Regression from scikit-learn.
    tagged_table : TaggedTable
        The tagged table containing the feature and target vectors.

    Raises
    ------
    LearningError
        If the tagged table contains invalid values or if the training failed.
    """
    try:
        model.fit(
            tagged_table.features._data,
            tagged_table.target._data,
        )
    except ValueError as exception:
        raise LearningError(str(exception)) from exception
    except Exception as exception:
        raise LearningError(None) from exception


# noinspection PyProtectedMember
def predict(model: Any, dataset: Table, target_name: Optional[str]) -> TaggedTable:
    """
    Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

    Parameters
    ----------
    model
        Classifier or regressor from scikit-learn.
    dataset : Table
        The dataset containing the features.
    target_name : str
        The name of the target column.

    Returns
    -------
    table : TaggedTable
        A dataset containing the given features and the predicted target.

    Raises
    ------
    PredictionError
        If predicting with the given dataset failed.
    """

    if model is None or target_name is None:
        raise PredictionError("The model was not trained")

    dataset_df = dataset._data
    dataset_df.columns = dataset.schema.get_column_names()
    try:
        predicted_target_vector = model.predict(dataset_df.values)
        result_set = dataset_df.copy(deep=True)
        if target_name in result_set.columns:
            raise ValueError(
                f"Dataset already contains '{target_name}' column. Please rename this column"
            )
        result_set[target_name] = predicted_target_vector
        return TaggedTable(Table(result_set), target_name=target_name)
    except NotFittedError as exception:
        raise PredictionError("The model was not trained") from exception
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
    except Exception as exception:
        raise PredictionError(None) from exception
