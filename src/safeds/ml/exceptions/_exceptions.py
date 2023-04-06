class DatasetContainsTargetError(ValueError):
    """
    Raised when a dataset contains the target column already.

    Parameters
    ----------
    target_name: str
        The name of the target column.
    """

    def __init__(self, target_name: str):
        super().__init__(f"Dataset already contains the target column '{target_name}'.")


class DatasetMissesFeaturesError(ValueError):
    """
    Raised when a dataset misses feature columns.

    Parameters
    ----------
    missing_feature_names: list[str]
        The names of the missing feature columns.
    """

    def __init__(self, missing_feature_names: list[str]):
        super().__init__(f"Dataset misses the feature columns '{missing_feature_names}'.")


class LearningError(Exception):
    """
    Raised when an error occurred while training a model.

    Parameters
    ----------
    reason: str | None
        The reason for the error.
    """

    def __init__(self, reason: str | None):
        if reason is None:
            super().__init__("Error occurred while learning")
        else:
            super().__init__(f"Error occurred while learning: {reason}")


class ModelNotFittedError(Exception):
    """Raised when a model is used before fitting it."""

    def __init__(self) -> None:
        super().__init__("The model has not been fitted yet.")


class PredictionError(Exception):
    """
    Raised when an error occurred while prediction a target vector using a model.

    Parameters
    ----------
    reason: str | None
        The reason for the error.
    """

    def __init__(self, reason: str | None):
        if reason is None:
            super().__init__("Error occurred while predicting")
        else:
            super().__init__(f"Error occurred while predicting: {reason}")
