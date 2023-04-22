import pytest

from safeds.data.tabular.containers import Column


class TestReprHtml:
    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_table_element(self, column: Column) -> None:
        assert "<table" in column._repr_html_()
