from safeds.data import SupervisedDataset, Table


def test_supervised_dataset_feature_vectors() -> None:
    table = Table.from_csv("tests/resources/test_supervised_dataset.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    assert "T" not in supervised_dataset.feature_vectors._data
    assert supervised_dataset.feature_vectors.schema.has_column("A")
    assert supervised_dataset.feature_vectors.schema.has_column("B")
    assert supervised_dataset.feature_vectors.schema.has_column("C")
