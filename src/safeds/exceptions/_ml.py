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


class DatasetMissesDataError(ValueError):
    """Raised when a dataset contains no rows."""

    def __init__(self) -> None:
        super().__init__("Dataset contains no rows")


class LearningError(Exception):
    """
    Raised when an error occurred while training a model.

    Parameters
    ----------
    reason: str
        The reason for the error.
    """

    def __init__(self, reason: str):
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
    reason: str
        The reason for the error.
    """

    def __init__(self, reason: str):
        super().__init__(f"Error occurred while predicting: {reason}")


class FeatureDataMismatchError(Exception):
    """Raised when the columns of the table passed to the predict or fit method do not match with the specified features of the neural network."""

    def __init__(self) -> None:
        super().__init__(
            "The features in the given table do not match with the specified feature columns names of the neural network.",
        )


class InputSizeError(Exception):
    """Raised when the amount of features being passed to a network does not match with its input size."""

    def __init__(self, table_size: int, input_layer_size: int) -> None:
        super().__init__(
            f"The amount of columns being passed to the network({table_size}) does not match with its input size({input_layer_size}). Consider changing the number of neurons in the first layer or reformatting the table.",
        )


class UntaggedTableError(Exception):
    """Raised when an untagged table is used instead of a TaggedTable in a regression or classification."""

    def __init__(self) -> None:
        super().__init__(
            (
                "This method needs a tagged table.\nA tagged table is a table that additionally knows which columns are"
                " features and which are the target to predict.\nUse Table.tag_column() to create a tagged table."
            ),
        )


class NonTimeSeriesError(Exception):
    """Raised when a table is used instead of a TimeSeries in a regression or classification."""

    def __init__(self) -> None:
        super().__init__(
            (
                "This method needs a time series.\nA time series is a table that additionally knows which columns are"
                " time and which are the target to predict.\n"
            ),
        )
