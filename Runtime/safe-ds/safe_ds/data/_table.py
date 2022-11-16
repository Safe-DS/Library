import pandas as pd


class Table:
    def __init__(self, data: pd.DataFrame):
        self._data: pd.DataFrame = data
