import pandas as pd


class Row:
    data: pd.Series

    def __init__(self, data: pd.Series):
        self.data = data
