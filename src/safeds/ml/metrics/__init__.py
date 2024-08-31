"""Classes to evaluate the performance of machine learning models."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._classification_metrics import ClassificationMetrics
    from ._classifier_metric import ClassifierMetric
    from ._regression_metrics import RegressionMetrics
    from ._regressor_metric import RegressorMetric

apipkg.initpkg(
    __name__,
    {
        "ClassificationMetrics": "._classification_metrics:ClassificationMetrics",
        "RegressionMetrics": "._regression_metrics:RegressionMetrics",
        "RegressorMetric": "._regressor_metric:RegressorMetric",
        "ClassifierMetric": "._classifier_metric:ClassifierMetric",
    },
)

__all__ = [
    "ClassifierMetric",
    "ClassificationMetrics",
    "RegressorMetric",
    "RegressionMetrics",
]
