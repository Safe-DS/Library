from safe_ds.data import Column
from sklearn.metrics import accuracy_score


def accuracy(actual: Column, expected: Column) -> float:
    return accuracy_score(actual._data, expected._data)
