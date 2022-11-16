import pandas as pd


class Column:
    def __init__(self, data: pd.Series):
        self._data: pd.Series = data
