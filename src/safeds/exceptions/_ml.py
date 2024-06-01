from safeds.ml.nn.typing import ModelImageSize


class DatasetMissesFeaturesError(ValueError):
    """
    Raised when a dataset misses feature columns.

    Parameters
    ----------
    missing_feature_names:
        The names of the missing feature columns.
    """

    def __init__(self, missing_feature_names: list[str]):
        super().__init__(f"Dataset misses the feature columns '{missing_feature_names}'.")

class DatasetMissesTargetError(ValueError):
    """
    Raised when a dataset misses the target column.

    Parameters
    ----------
    missing_target_name:
        The names of the missing target column.
    """

    def __init__(self, missing_target_name: str):
        super().__init__(f"Dataset misses the target column '{missing_target_name}'.")


class DatasetMissesDataError(ValueError):
    """Raised when a dataset contains no rows."""

    def __init__(self) -> None:
        super().__init__("Dataset contains no rows")


class LearningError(Exception):
    """
    Raised when an error occurred while training a model.

    Parameters
    ----------
    reason:
        The reason for the error.
    """

    def __init__(self, reason: str):
        super().__init__(f"Error occurred while learning: {reason}")


class ModelNotFittedError(RuntimeError):
    """Raised when a model is used before fitting it."""

    def __init__(self) -> None:
        super().__init__("The model has not been fitted yet.")


class InvalidModelStructureError(Exception):
    """Raised when the structure of the model is invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"The model structure is invalid: {reason}")


class PredictionError(Exception):
    """
    Raised when an error occurred while prediction a target vector using a model.

    Parameters
    ----------
    reason:
        The reason for the error.
    """

    def __init__(self, reason: str):
        super().__init__(f"Error occurred while predicting: {reason}")


class FeatureDataMismatchError(Exception):
    """Raised when the columns of the table passed to the predict or fit method do not match with the specified features of the model."""

    def __init__(self) -> None:
        super().__init__(
            "The features in the given table do not match with the specified feature columns names of model.",
        )


class InputSizeError(Exception):
    """Raised when the amount of features being passed to a model does not match with its input size."""

    def __init__(self, data_size: int | ModelImageSize, input_layer_size: int | ModelImageSize | None) -> None:
        # TODO: remove input_layer_size type None again
        super().__init__(
            f"The data size being passed to the network({data_size}) does not match with its input size({input_layer_size}). Consider changing the data size of the model or reformatting the data.",
        )


class PlainTableError(TypeError):
    """Raised when a plain table is used instead of a TabularDataset."""

    def __init__(self) -> None:
        super().__init__(
            (
                "This method needs a tabular dataset. "
                "It knows which columns are features and which are the target to predict.\n"
                "Use `Table.to_tabular_dataset()` to create a tabular dataset."
            ),
        )
