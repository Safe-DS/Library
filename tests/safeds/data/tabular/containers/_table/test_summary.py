import pytest
from safeds.data.tabular.containers import Table
from statistics import stdev


@pytest.mark.parametrize(
    ("table", "truth"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}),
         Table.from_dict({
             "metrics": [
                 "maximum",
                 "minimum",
                 "mean",
                 "mode",
                 "median",
                 "sum",
                 "variance",
                 "standard deviation",
                 "idness",
                 "stability",
             ],
             "col1": [
                 "2",
                 "1",
                 str(4.0 / 3),
                 "[1]",
                 "1.0",
                 "4",
                 str(1.0 / 3),
                 str(stdev([1, 2, 1])),
                 str(2.0 / 3),
                 str(2.0 / 3),
             ],
             "col2": [
                 "-",
                 "-",
                 "-",
                 "['a', 'b', 'c']",
                 "-",
                 "-",
                 "-",
                 "-",
                 "1.0",
                 str(1.0 / 3),
             ],
         },
         )
         )
    ]
)
def test_should_make_summary(table, truth) -> None:
    assert truth == table.summary()
