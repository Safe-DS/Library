class LearningError(Exception):
    """Exception raised when an error occurred while training a model.

    Parameters
    ----------
    reason: str | None
        Reason of the Error
    """

    def __init__(self, reason: str | None):
        if reason is None:
            super().__init__("Error occurred while learning")
        else:
            super().__init__(f"Error occurred while learning: {reason}")


class PredictionError(Exception):
    """Exception raised when an error occurred while prediction a target vector using a model.

    Parameters
    ----------
    reason: str | None
        Reason of the Error
    """

    def __init__(self, reason: str | None):
        if reason is None:
            super().__init__("Error occurred while predicting")
        else:
            super().__init__(f"Error occurred while predicting: {reason}")
