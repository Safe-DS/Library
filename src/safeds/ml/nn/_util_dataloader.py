from safeds.data.tabular.containers import Column
import numpy as np

def _window_column(win_len: int, col: Column) -> (np.array, Column):
    """Takes a Column and applies windows to it, so the first column of the tuple contains the features and the second
     the target"""
    seq = col._data.to_numpy()
    x_s = []
    y_s = []
    L = len(seq)
    for i in range(L-win_len*2):
        window = seq[i:i+win_len]
        label = seq[i+win_len:i+win_len+win_len]
        x_s.append(window)
        y_s.append(label)
    print(len(x_s))
    print(len(y_s))
    return np.array(x_s), np.array(y_s)
