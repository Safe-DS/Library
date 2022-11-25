from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TableSchema:
    """Stores column names and corresponding data types for a table

    Parameters
    ----------
    column_names: list[str]
        Column names as an array
    data_types: list[numpy.dtype]
        Dataypes as an array using the numpy dtpye class

    """

    _schema: OrderedDict

    def __init__(self, column_names: list[str], data_types: list[np.dtype]):
        self._schema = OrderedDict()
        for column_name, data_type in zip(column_names, data_types):
            self._schema[column_name] = data_type

    def has_column(self, column_name: str) -> bool:
        return column_name in self._schema

    def get_type_of_column(self, column_name: str) -> Optional[np.dtype]:
        return self._schema[column_name]
