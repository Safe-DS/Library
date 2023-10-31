import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


#definiere LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out



def test_create_timeseries() -> None:
    table = Table(data={"f1": [1, 2, 3, 4, 6, 7], "target": [7,2, 3, 1, 3, 7], "f2": [4,7, 5, 5, 5, 7]})
    ts = TimeSeries(data={"f1": [1, 2, 3, 4, 6, 7], "target": [7,2, 3, 1, 3, 7], "f2": [4,7, 5, 5, 5, 7]},
                    target_name="target",
                    date_name="f1",
                    window_size=2,
                    forecast_horizon=1,
                    feature_names=["f1", "f2", "target"])
    
    
    # ein Modell erstellen ist in safeDS noch nicht definiert darum low level in PyTorch
    # 2 ist hier die number der feature Columns
    input_dim = ts._window_size * len(ts._feature_names)
    hidden_dim = 1
    output_dim = ts._forecast_horizon
    model = LSTMModel(input_dim, hidden_dim, output_dim)

    #damit der Datensatz low level laden kann hier into_Dataloader
    loader = ts.into_DataLoader()

    for batch in iter(loader):
        inputs, labels = batch
        inputs = inputs.to(torch.float32)
        model(inputs)
    

    #wenn durchläuft wurde korrekt Table in Dataloader geladen
    #assert False
