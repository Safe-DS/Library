import pandas as pd


class Column:
    data: pd.Series

    def __init__(self, data: pd.Series):
        self.data = data
