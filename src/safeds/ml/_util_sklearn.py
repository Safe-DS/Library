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


# noinspection PyProtectedMember
def predict(model: Any, dataset: Table, feature_names: Optional[list[str]], target_name: Optional[str]) -> TaggedTable:
    """
    Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

    Parameters
    ----------
    model
        Classifier or regressor from scikit-learn.
    dataset : Table
        The dataset containing the features.
    target_name : Optional[str]
        The name of the target column.
    feature_names : Optional[list[str]]
        The names of the feature columns.

    Returns
    -------
    table : TaggedTable
        A dataset containing the given features and the predicted target.

    Raises
    ------
    PredictionError
        If predicting with the given dataset failed.
    """

    if model is None or target_name is None or feature_names is None:
        raise PredictionError("The model was not trained")

    dataset_df = dataset.keep_only_columns(feature_names)._data
    dataset_df.columns = feature_names
    try:
        predicted_target_vector = model.predict(dataset_df.values)
        result_set = dataset._data.copy(deep=True)
        result_set.columns = dataset.get_column_names()
        if target_name in result_set.columns:
            raise ValueError(f"Dataset already contains '{target_name}' column. Please rename this column")
        result_set[target_name] = predicted_target_vector
        return Table(result_set).tag_columns(target_name=target_name, feature_names=feature_names)
    except NotFittedError as exception:
        raise PredictionError("The model was not trained") from exception
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
