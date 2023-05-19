from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import TableTransformer, InvertibleTableTransformer
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError


class RangeScaler(InvertibleTableTransformer):
    """
    Normalize Values in a Table

    Parameters
    ----------

    Examples
    --------
    """

    def __init__(self, minimum:float = 0.0 , maximum:float = 1.0):
        self._minimum = minimum
        self._maximum = maximum


