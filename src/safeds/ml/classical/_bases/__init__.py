"""Base classes for classical machine learning models."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._ada_boost_base import _AdaBoostBase
    from ._decision_tree_base import _DecisionTreeBase
    from ._gradient_boosting_base import _GradientBoostingBase
    from ._k_nearest_neighbors_base import _KNearestNeighborsBase
    from ._random_forest_base import _RandomForestBase
    from ._support_vector_machine_base import _SupportVectorMachineBase

apipkg.initpkg(
    __name__,
    {
        "_AdaBoostBase": "._ada_boost_base:_AdaBoostBase",
        "_DecisionTreeBase": "._decision_tree_base:_DecisionTreeBase",
        "_GradientBoostingBase": "._gradient_boosting_base:_GradientBoostingBase",
        "_KNearestNeighborsBase": "._k_nearest_neighbors_base:_KNearestNeighborsBase",
        "_RandomForestBase": "._random_forest_base:_RandomForestBase",
        "_SupportVectorMachineBase": "._support_vector_machine_base:_SupportVectorMachineBase",
    },
)

__all__ = [
    "_AdaBoostBase",
    "_DecisionTreeBase",
    "_GradientBoostingBase",
    "_KNearestNeighborsBase",
    "_RandomForestBase",
    "_SupportVectorMachineBase",
]
