from __future__ import annotations


class RegressionMetrics:

    @staticmethod
    def mean_squared_error(expected: Column | TabularDataset, predicted: Column | TabularDataset):
        return mean_squared_error(expected, predicted)

    @staticmethod
    def mean_absolute_error(expected: Column | TabularDataset, predicted: Column | TabularDataset):
        return mean_absolute_error(expected, predicted)

    @staticmethod
    def r2_score(expected: Column | TabularDataset, predicted: Column | TabularDataset):
        return r2_score(expected, predicted)

    @staticmethod
    def summarize(expected: Column | TabularDataset, predicted: Column | TabularDataset) -> Table:
        mse = RegressionMetrics.mean_squared_error(expected, predicted)
        mae = RegressionMetrics.mean_absolute_error(expected, predicted)
        r2 = RegressionMetrics.r2_score(expected, predicted)
        return {'mse': mse, 'mae': mae, 'r2': r2}
