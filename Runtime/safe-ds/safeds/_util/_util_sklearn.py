from typing import Any

from safeds.data import SupervisedDataset, Table
from safeds.exceptions import LearningError, PredictionError
from sklearn.exceptions import NotFittedError


# noinspection PyProtectedMember
def fit(model: Any, supervised_dataset: SupervisedDataset) -> str:
    """
    Fit a model for a given supervised dataset.

    Parameters
    ----------
    model
        Classifier or Regression from scikit-learn.
    supervised_dataset : SupervisedDataset
        The supervised dataset containing the feature and target vectors.

    Returns
    -------
    target_name : str
        The target column name, inferred from the supervised dataset.

    Raises
    ------
    LearningError
        If the supervised dataset contains invalid values or if the training failed.
    """
    try:
        model.fit(
            supervised_dataset.feature_vectors._data,
            supervised_dataset.target_values._data,
        )
        return supervised_dataset.target_values.name
    except ValueError as exception:
        raise LearningError(str(exception)) from exception
    except Exception as exception:
        raise LearningError(None) from exception


# noinspection PyProtectedMember
def predict(model: Any, dataset: Table, target_name: str) -> Table:
    """
    Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

    Parameters
    ----------
    model
        Classifier or Regression from scikit-learn.
    dataset : Table
        The dataset containing the feature vectors.
    target_name : str
        The name of the target vector, the name of the target column inferred from fit is used by default.

    Returns
    -------
    table : Table
        A dataset containing the given feature vectors and the predicted target vector.

    Raises
    ------
    PredictionError
        If predicting with the given dataset failed.
    """
    dataset_df = dataset._data
    dataset_df.columns = dataset.schema.get_column_names()
    try:
        predicted_target_vector = model.predict(dataset_df)
        result_set = dataset_df.copy(deep=True)
        if target_name in result_set.columns:
            raise ValueError(
                f"Dataset already contains '{target_name}' column. Please rename this column"
            )
        result_set[target_name] = predicted_target_vector
        return Table(result_set)
    except NotFittedError as exception:
        raise PredictionError("The model was not trained") from exception
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
    except Exception as exception:
        raise PredictionError(None) from exception
