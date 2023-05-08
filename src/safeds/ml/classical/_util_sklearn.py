import warnings
from typing import Any

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import (
    DatasetContainsTargetError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
    UntaggedTableError,
)


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
    UntaggedTableError
        If the table is untagged.
    """
    if not isinstance(tagged_table, TaggedTable) and isinstance(tagged_table, Table):
        raise UntaggedTableError
    try:
        model.fit(
            tagged_table.features._data,
            tagged_table.target._data,
        )
    except ValueError as exception:
        raise LearningError(str(exception)) from exception


# noinspection PyProtectedMember
def predict(model: Any, dataset: Table, feature_names: list[str] | None, target_name: str | None) -> TaggedTable:
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
    ModelNotFittedError
        If the model has not been fitted yet.
    DatasetContainsTargetError
        If the dataset contains the target column already.
    DatasetMissesFeaturesError
        If the dataset misses feature columns.
    PredictionError
        If predicting with the given dataset failed.
    """
    # Validation
    if model is None or target_name is None or feature_names is None:
        raise ModelNotFittedError
    if dataset.has_column(target_name):
        raise DatasetContainsTargetError(target_name)
    missing_feature_names = [feature_name for feature_name in feature_names if not dataset.has_column(feature_name)]
    if missing_feature_names:
        raise DatasetMissesFeaturesError(missing_feature_names)

    dataset_df = dataset.keep_only_columns(feature_names)._data
    dataset_df.columns = feature_names

    result_set = dataset._data.copy(deep=True)
    result_set.columns = dataset.column_names

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            predicted_target_vector = model.predict(dataset_df.values)
        result_set[target_name] = predicted_target_vector
        return Table._from_pandas_dataframe(result_set).tag_columns(
            target_name=target_name,
            feature_names=feature_names,
        )
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
