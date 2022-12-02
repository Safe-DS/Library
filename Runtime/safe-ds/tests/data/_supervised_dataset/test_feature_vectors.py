from safe_ds.data import SupervisedDataset, Table


def test_supervised_dataset_feature_vectors():
    table = Table.from_csv("tests/resources/test_supervised_dataset.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    assert "T" not in supervised_dataset.feature_vectors._data
    assert "A" in supervised_dataset.feature_vectors._data
    assert "B" in supervised_dataset.feature_vectors._data
    assert "C" in supervised_dataset.feature_vectors._data
