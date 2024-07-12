"""Classes for transforming tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._discretizer import Discretizer
    from ._functional_table_transformer import FunctionalTableTransformer
    from ._invertible_table_transformer import InvertibleTableTransformer
    from ._k_nearest_neighbors_imputer import KNearestNeighborsImputer
    from ._label_encoder import LabelEncoder
    from ._one_hot_encoder import OneHotEncoder
    from ._range_scaler import RangeScaler
    from ._robust_scaler import RobustScaler
    from ._sequential_table_transformer import SequentialTableTransformer
    from ._simple_imputer import SimpleImputer
    from ._standard_scaler import StandardScaler
    from ._table_transformer import TableTransformer


apipkg.initpkg(
    __name__,
    {
        "Discretizer": "._discretizer:Discretizer",
        "FunctionalTableTransformer": "._functional_table_transformer:FunctionalTableTransformer",
        "InvertibleTableTransformer": "._invertible_table_transformer:InvertibleTableTransformer",
        "LabelEncoder": "._label_encoder:LabelEncoder",
        "OneHotEncoder": "._one_hot_encoder:OneHotEncoder",
        "RangeScaler": "._range_scaler:RangeScaler",
        "SequentialTableTransformer": "._sequential_table_transformer:SequentialTableTransformer",
        "RobustScaler": "._robust_scaler:RobustScaler",
        "SimpleImputer": "._simple_imputer:SimpleImputer",
        "StandardScaler": "._standard_scaler:StandardScaler",
        "TableTransformer": "._table_transformer:TableTransformer",
        "KNearestNeighborsImputer": "._k_nearest_neighbors_imputer:KNearestNeighborsImputer",
    },
)

__all__ = [
    "Discretizer",
    "FunctionalTableTransformer",
    "InvertibleTableTransformer",
    "LabelEncoder",
    "OneHotEncoder",
    "RangeScaler",
    "SequentialTableTransformer",
    "RobustScaler",
    "SimpleImputer",
    "StandardScaler",
    "TableTransformer",
    "KNearestNeighborsImputer",
]
