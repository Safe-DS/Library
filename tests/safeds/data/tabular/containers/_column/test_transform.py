from typing import Mapping

from safeds.data.tabular.containers import Column, Row


def test_should_transform_column():
    column = Column("test", [1, 2])
    column = column.transform(lambda it: it + 1)

    assert column[0] == 2
    assert column[1] == 3
