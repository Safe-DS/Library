import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError
from safeds.ml.nn import RNN_Layer, Model




def test_create_timeseries() -> None:
    table = Table.from_csv_file(r"tests\resources\Alcohol_Sales (1).csv")
    ts = TimeSeries._from_table(table,target_name="S4248SM144NCEN", date_name="DATE", window_size=12, forecast_horizon=1, feature_names=["S4248SM144NCEN"])
    
    
    
    # ein Modell erstellen ist in safeDS noch nicht definiert darum low level in PyTorch
    # 2 ist hier die number der feature Columns
    input_dim = ts._window_size * len(ts._feature_names)
    hidden_dim = 256
    output_dim = ts._forecast_horizon
    layer1 = RNN_Layer(input_dim, hidden_dim) 
    layer2 = RNN_Layer(hidden_dim, output_dim)
    model = Model([layer1, layer2])
    #model.train(ts.into_DataLoader(), 5, 0.01)

    


    

    #wenn durchläuft wurde korrekt Table in Dataloader geladen
    assert False


