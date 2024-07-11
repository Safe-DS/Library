"""Classes to evaluate the performance of machine learning models."""

from typing import TYPE_CHECKING

import apipkg

from ._classifier_metric import ClassifierMetric
from ._regressor_metric import RegressorMetric

if TYPE_CHECKING:
    from ._classification_metrics import ClassificationMetrics
    from ._regression_metrics import RegressionMetrics

apipkg.initpkg(
    __name__,
    {
        "ClassificationMetrics": "._classification_metrics:ClassificationMetrics",
        "ClassifierMetric": "._classifier_metrics:ClassifierMetric",
        "RegressionMetrics": "._regression_metrics:RegressionMetrics",
        "RegressorMetric": "._regressor_metric:RegressorMetric",
    },
)

__all__ = [
    "ClassificationMetrics",
    "ClassifierMetric",
    "RegressionMetrics",
    "RegressorMetric",
]
