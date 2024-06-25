from enum import Enum


class ClassifierMetric(Enum):
    """An Enum of possible Metrics for a Classifier."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
