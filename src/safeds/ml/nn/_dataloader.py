import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from safeds.data.tabular.containers import TaggedTable


class CustomDataloader(DataLoader):
    def __init__(self, table: TaggedTable, batch_size):
        self._loader = DataLoader(dataset=CustomDataset(table), batch_size=batch_size, shuffle=True)
        pass


class CustomDataset(Dataset):
    def __init__(self, table: TaggedTable):
        # TODO Create a tensor from a tagged Table correctly
        self.X = torch.from_numpy(table.remove_columns([table.target]).astype(np.float32))
        self.Y = torch.from_numpy(table.features.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.len
