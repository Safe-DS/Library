import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import UnknownColumnNameError

def test_should_warn_if_similar_column_name() -> None:
    table1 = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with pytest.warns(
        UserWarning,
        match=(
            f"did you mean col1?"
        ),
    ):
        table1.get_similar_columns("cil1")
