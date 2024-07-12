import pytest

from safeds.data.tabular.transformation import ParallelTableTransformer, RobustScaler, StandardScaler

#l = ParallelTableTransformer(transformers=[RobustScaler(column_names=["a", "b"]), RangeScaler(column_names=["c", "d"])])
#t = Table.from_dict({
#    "a": [1, 2, 3],
#    "b": [4, 5, 6],
#    "c": [7, 8, 9],
#    "d": [9, 10, 11],
#})
#fl = l.fit(t)
#tt = fl.transform(t)
#print(tt)"""

class TestInit:
    def test_should_raise_if_no_transformers_were_supplied():
        with pytest.raises(ValueError, match="Transformers must contain at least one transformer."):
            ParallelTableTransformer(transformers=[])

    def test_should_raise_if_columns_overlap():
        with pytest.raises(ValueError, match="Cannot apply two transformers to the same column at the same time."):
            ParallelTableTransformer(transformers=[RobustScaler("a"), RobustScaler("a")])

    def test_should_pass():
        tf = ParallelTableTransformer(transformers=[RobustScaler("a"), StandardScaler("b")])

class TestFit:
    def test_should_pass():
        tf = ParallelTableTransformer(transformers=[RobustScaler(column_names=["a", "b"]), RangeScaler(column_names=["c", "d"])])
        t = Table.from_dict({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [9, 10, 11],
        })
        ftf = tf.fit(t)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        Table(
            {
                "a": [1, 2, 3],
                "b": [2, 3, 4],
                "c": [3, 4, 5],
                "d": [4, 5, 6],
            },
        ),
        Table(
            {
                "a": [1, 2, 3],
                "b": ["a", "b", "c"],
                "c": [1, 2, 3],
                "d": [None, 1, 4],
            },
        ),
        Table(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [0.9, 2.4, 99.0],
            },
        ),
    ],
    ids=[
        "numeric only",
        "numeric - non-numeric mixed",
        "missing column d",
    ],
)
class TestTransform:
    def test_should_pass():
        pass
    def test_should_raise_if

class TestInverseTransform: